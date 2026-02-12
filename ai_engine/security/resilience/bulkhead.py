"""
QBITEL Engine - Bulkhead Isolation Implementation

Enterprise-grade bulkhead pattern to isolate resources and prevent
cascading failures in the Zero-Touch Security Orchestrator.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import weakref

from ..logging import get_security_logger, SecurityLogType, LogLevel


class ResourceState(str, Enum):
    """State of a bulkhead resource."""

    AVAILABLE = "available"
    BUSY = "busy"
    EXHAUSTED = "exhausted"
    DEGRADED = "degraded"


class AllocationStrategy(str, Enum):
    """Strategy for allocating resources."""

    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    ROUND_ROBIN = "round_robin"  # Round-robin allocation
    PRIORITY = "priority"  # Priority-based allocation


@dataclass
class ResourceMetrics:
    """Metrics for resource usage."""

    total_capacity: int = 0
    available_capacity: int = 0
    allocated_capacity: int = 0
    pending_requests: int = 0
    total_requests: int = 0
    successful_allocations: int = 0
    failed_allocations: int = 0
    timeout_allocations: int = 0
    average_wait_time: float = 0.0
    max_wait_time: float = 0.0
    utilization_rate: float = 0.0

    def update_utilization(self):
        """Update utilization rate."""
        if self.total_capacity > 0:
            self.utilization_rate = self.allocated_capacity / self.total_capacity
        else:
            self.utilization_rate = 0.0


@dataclass
class AllocationRequest:
    """Request for resource allocation."""

    request_id: str
    priority: int = 0
    timeout: Optional[float] = None
    requested_at: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.requested_at is None:
            self.requested_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class ResourcePool:
    """
    Manages a pool of resources with configurable allocation strategies
    and isolation boundaries.
    """

    def __init__(
        self,
        name: str,
        capacity: int,
        allocation_strategy: AllocationStrategy = AllocationStrategy.FIFO,
        timeout: float = 30.0,
        enable_degraded_mode: bool = True,
        degraded_capacity_threshold: float = 0.8,
        rejection_threshold: float = 0.95,
    ):
        self.name = name
        self.capacity = capacity
        self.allocation_strategy = allocation_strategy
        self.timeout = timeout
        self.enable_degraded_mode = enable_degraded_mode
        self.degraded_capacity_threshold = degraded_capacity_threshold
        self.rejection_threshold = rejection_threshold

        self.logger = get_security_logger(f"qbitel.security.resilience.bulkhead.{name}")

        # Resource management
        self._semaphore = asyncio.Semaphore(capacity)
        self._available_resources = list(range(capacity))
        self._allocated_resources: Dict[str, int] = {}
        self._pending_queue: List[AllocationRequest] = []
        self._round_robin_index = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._metrics = ResourceMetrics(total_capacity=capacity, available_capacity=capacity)
        self._allocation_times: List[float] = []

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Resource pool '{name}' initialized with capacity {capacity}",
            level=LogLevel.INFO,
            metadata={
                "capacity": capacity,
                "allocation_strategy": allocation_strategy.value,
                "timeout": timeout,
            },
        )

    @property
    def state(self) -> ResourceState:
        """Get current resource pool state."""
        utilization = self._metrics.utilization_rate

        if utilization >= self.rejection_threshold:
            return ResourceState.EXHAUSTED
        elif utilization >= self.degraded_capacity_threshold:
            return ResourceState.DEGRADED
        elif self._metrics.allocated_capacity > 0:
            return ResourceState.BUSY
        else:
            return ResourceState.AVAILABLE

    def get_metrics(self) -> Dict[str, Any]:
        """Get resource pool metrics."""
        self._metrics.update_utilization()

        return {
            "name": self.name,
            "state": self.state.value,
            "capacity": self.capacity,
            "allocation_strategy": self.allocation_strategy.value,
            "metrics": {
                "total_capacity": self._metrics.total_capacity,
                "available_capacity": self._metrics.available_capacity,
                "allocated_capacity": self._metrics.allocated_capacity,
                "pending_requests": self._metrics.pending_requests,
                "total_requests": self._metrics.total_requests,
                "successful_allocations": self._metrics.successful_allocations,
                "failed_allocations": self._metrics.failed_allocations,
                "timeout_allocations": self._metrics.timeout_allocations,
                "average_wait_time": self._metrics.average_wait_time,
                "max_wait_time": self._metrics.max_wait_time,
                "utilization_rate": self._metrics.utilization_rate,
            },
        }

    async def acquire(
        self,
        request_id: str,
        priority: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """
        Acquire a resource from the pool.

        Args:
            request_id: Unique identifier for the request
            priority: Priority level (higher = more priority)
            timeout: Optional timeout override
            metadata: Additional request metadata

        Returns:
            Resource ID if successful, None if failed
        """

        request_timeout = timeout or self.timeout
        request = AllocationRequest(
            request_id=request_id,
            priority=priority,
            timeout=request_timeout,
            metadata=metadata or {},
        )

        start_time = asyncio.get_event_loop().time()

        async with self._lock:
            self._metrics.total_requests += 1
            self._metrics.pending_requests += 1

        try:
            # Check if we should reject immediately due to exhaustion
            if self.state == ResourceState.EXHAUSTED:
                async with self._lock:
                    self._metrics.failed_allocations += 1
                    self._metrics.pending_requests -= 1

                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Resource pool '{self.name}' rejected request - pool exhausted",
                    level=LogLevel.WARNING,
                    metadata={
                        "request_id": request_id,
                        "utilization_rate": self._metrics.utilization_rate,
                        "state": self.state.value,
                    },
                )

                return None

            # Wait for semaphore with timeout
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=request_timeout)
            except asyncio.TimeoutError:
                async with self._lock:
                    self._metrics.timeout_allocations += 1
                    self._metrics.failed_allocations += 1
                    self._metrics.pending_requests -= 1

                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Resource pool '{self.name}' allocation timeout",
                    level=LogLevel.WARNING,
                    metadata={
                        "request_id": request_id,
                        "timeout": request_timeout,
                        "wait_time": asyncio.get_event_loop().time() - start_time,
                    },
                )

                return None

            # Allocate resource
            async with self._lock:
                resource_id = await self._allocate_resource(request)

                if resource_id is not None:
                    wait_time = asyncio.get_event_loop().time() - start_time
                    self._allocation_times.append(wait_time)

                    # Update metrics
                    self._metrics.successful_allocations += 1
                    self._metrics.pending_requests -= 1
                    self._metrics.allocated_capacity += 1
                    self._metrics.available_capacity -= 1

                    # Update wait time metrics
                    if len(self._allocation_times) > 0:
                        self._metrics.average_wait_time = sum(self._allocation_times) / len(self._allocation_times)
                        self._metrics.max_wait_time = max(self._allocation_times)

                        # Keep only recent allocation times for rolling average
                        if len(self._allocation_times) > 1000:
                            self._allocation_times = self._allocation_times[-500:]

                    self.logger.log_security_event(
                        SecurityLogType.PERFORMANCE_METRIC,
                        f"Resource allocated from pool '{self.name}'",
                        level=LogLevel.DEBUG,
                        metadata={
                            "request_id": request_id,
                            "resource_id": resource_id,
                            "wait_time": wait_time,
                            "utilization_rate": self._metrics.utilization_rate,
                        },
                    )
                else:
                    # Release semaphore if allocation failed
                    self._semaphore.release()
                    self._metrics.failed_allocations += 1
                    self._metrics.pending_requests -= 1

                return resource_id

        except Exception as e:
            async with self._lock:
                self._metrics.failed_allocations += 1
                self._metrics.pending_requests -= 1

            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Resource pool '{self.name}' allocation error: {str(e)}",
                level=LogLevel.ERROR,
                metadata={"request_id": request_id, "exception": str(e)},
                error_code="RESOURCE_ALLOCATION_ERROR",
            )

            raise

    async def _allocate_resource(self, request: AllocationRequest) -> Optional[int]:
        """Allocate a resource using the configured strategy."""

        if not self._available_resources:
            return None

        if self.allocation_strategy == AllocationStrategy.FIFO:
            resource_id = self._available_resources.pop(0)
        elif self.allocation_strategy == AllocationStrategy.LIFO:
            resource_id = self._available_resources.pop()
        elif self.allocation_strategy == AllocationStrategy.ROUND_ROBIN:
            if self._round_robin_index >= len(self._available_resources):
                self._round_robin_index = 0
            resource_id = self._available_resources.pop(self._round_robin_index)
            if self._round_robin_index >= len(self._available_resources):
                self._round_robin_index = 0
        elif self.allocation_strategy == AllocationStrategy.PRIORITY:
            # For priority allocation, we would need to implement priority queuing
            # For now, use FIFO as fallback
            resource_id = self._available_resources.pop(0)
        else:
            resource_id = self._available_resources.pop(0)

        self._allocated_resources[request.request_id] = resource_id
        return resource_id

    async def release(self, request_id: str) -> bool:
        """
        Release a resource back to the pool.

        Args:
            request_id: Request ID that was used to acquire the resource

        Returns:
            True if released successfully, False otherwise
        """

        async with self._lock:
            if request_id not in self._allocated_resources:
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Attempt to release unknown resource in pool '{self.name}'",
                    level=LogLevel.WARNING,
                    metadata={"request_id": request_id},
                )
                return False

            resource_id = self._allocated_resources.pop(request_id)
            self._available_resources.append(resource_id)

            # Update metrics
            self._metrics.allocated_capacity -= 1
            self._metrics.available_capacity += 1

            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Resource released to pool '{self.name}'",
                level=LogLevel.DEBUG,
                metadata={
                    "request_id": request_id,
                    "resource_id": resource_id,
                    "available_capacity": self._metrics.available_capacity,
                },
            )

        # Release semaphore permit
        self._semaphore.release()
        return True

    async def resize(self, new_capacity: int) -> bool:
        """
        Resize the resource pool capacity.

        Args:
            new_capacity: New capacity for the pool

        Returns:
            True if resized successfully, False otherwise
        """

        if new_capacity <= 0:
            return False

        async with self._lock:
            old_capacity = self.capacity

            if new_capacity > old_capacity:
                # Expand pool
                additional_resources = list(range(old_capacity, new_capacity))
                self._available_resources.extend(additional_resources)

                # Update semaphore
                for _ in range(new_capacity - old_capacity):
                    self._semaphore.release()

            elif new_capacity < old_capacity:
                # Shrink pool - only if we have enough available resources
                needed_reduction = old_capacity - new_capacity

                if len(self._available_resources) < needed_reduction:
                    # Cannot shrink - too many resources in use
                    self.logger.log_security_event(
                        SecurityLogType.CONFIGURATION_CHANGE,
                        f"Cannot shrink pool '{self.name}' - insufficient available resources",
                        level=LogLevel.WARNING,
                        metadata={
                            "current_capacity": old_capacity,
                            "requested_capacity": new_capacity,
                            "available_resources": len(self._available_resources),
                            "needed_reduction": needed_reduction,
                        },
                    )
                    return False

                # Remove resources from available list
                for _ in range(needed_reduction):
                    if self._available_resources:
                        self._available_resources.pop()

                # Acquire semaphore permits to reduce capacity
                for _ in range(needed_reduction):
                    try:
                        self._semaphore.acquire_nowait()
                    except ValueError:
                        # Semaphore already at capacity, this shouldn't happen
                        pass

            # Update capacity and metrics
            self.capacity = new_capacity
            self._metrics.total_capacity = new_capacity
            self._metrics.available_capacity = len(self._available_resources)
            self._metrics.update_utilization()

            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Resource pool '{self.name}' resized from {old_capacity} to {new_capacity}",
                level=LogLevel.INFO,
                metadata={
                    "old_capacity": old_capacity,
                    "new_capacity": new_capacity,
                    "available_capacity": self._metrics.available_capacity,
                    "allocated_capacity": self._metrics.allocated_capacity,
                },
            )

            return True

    def reset_metrics(self):
        """Reset pool metrics."""
        self._metrics = ResourceMetrics(
            total_capacity=self.capacity,
            available_capacity=len(self._available_resources),
        )
        self._allocation_times.clear()


class BulkheadManager:
    """
    Manages multiple resource pools implementing the bulkhead pattern
    for fault isolation and resource management.
    """

    def __init__(self):
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.logger = get_security_logger("qbitel.security.resilience.bulkhead_manager")

        # Global metrics
        self._total_pools = 0
        self._total_capacity = 0
        self._total_allocated = 0

    def create_pool(
        self,
        name: str,
        capacity: int,
        allocation_strategy: AllocationStrategy = AllocationStrategy.FIFO,
        **kwargs,
    ) -> ResourcePool:
        """
        Create a new resource pool.

        Args:
            name: Name of the resource pool
            capacity: Pool capacity
            allocation_strategy: Resource allocation strategy
            **kwargs: Additional pool configuration

        Returns:
            Created resource pool
        """

        if name in self.resource_pools:
            raise ValueError(f"Resource pool '{name}' already exists")

        pool = ResourcePool(
            name=name,
            capacity=capacity,
            allocation_strategy=allocation_strategy,
            **kwargs,
        )

        self.resource_pools[name] = pool
        self._total_pools += 1
        self._total_capacity += capacity

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Created resource pool: {name}",
            level=LogLevel.INFO,
            metadata={
                "pool_name": name,
                "capacity": capacity,
                "allocation_strategy": allocation_strategy.value,
                "total_pools": self._total_pools,
                "total_capacity": self._total_capacity,
            },
        )

        return pool

    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get resource pool by name."""
        return self.resource_pools.get(name)

    def list_pools(self) -> List[str]:
        """List all resource pool names."""
        return list(self.resource_pools.keys())

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global bulkhead metrics."""

        total_allocated = sum(pool._metrics.allocated_capacity for pool in self.resource_pools.values())

        total_available = sum(pool._metrics.available_capacity for pool in self.resource_pools.values())

        pool_states = {}
        for name, pool in self.resource_pools.items():
            pool_states[name] = pool.state.value

        return {
            "total_pools": self._total_pools,
            "total_capacity": self._total_capacity,
            "total_allocated": total_allocated,
            "total_available": total_available,
            "global_utilization": (total_allocated / self._total_capacity if self._total_capacity > 0 else 0),
            "pool_states": pool_states,
            "pool_metrics": {name: pool.get_metrics() for name, pool in self.resource_pools.items()},
        }

    async def acquire_from_pool(self, pool_name: str, request_id: str, **kwargs) -> Optional[int]:
        """Acquire resource from specified pool."""

        pool = self.get_pool(pool_name)
        if not pool:
            raise ValueError(f"Resource pool '{pool_name}' not found")

        return await pool.acquire(request_id, **kwargs)

    async def release_to_pool(self, pool_name: str, request_id: str) -> bool:
        """Release resource to specified pool."""

        pool = self.get_pool(pool_name)
        if not pool:
            raise ValueError(f"Resource pool '{pool_name}' not found")

        return await pool.release(request_id)

    async def shutdown(self):
        """Shutdown all resource pools."""

        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Shutting down {len(self.resource_pools)} resource pools",
            level=LogLevel.INFO,
        )

        # Clear all pools
        self.resource_pools.clear()
        self._total_pools = 0
        self._total_capacity = 0
        self._total_allocated = 0


# Context manager for resource acquisition
class ResourceContext:
    """Context manager for acquiring and releasing resources."""

    def __init__(self, pool: ResourcePool, request_id: str, **kwargs):
        self.pool = pool
        self.request_id = request_id
        self.kwargs = kwargs
        self.resource_id: Optional[int] = None

    async def __aenter__(self):
        """Acquire resource on context entry."""
        self.resource_id = await self.pool.acquire(self.request_id, **self.kwargs)
        return self.resource_id

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release resource on context exit."""
        if self.resource_id is not None:
            await self.pool.release(self.request_id)


# Global bulkhead manager instance
_bulkhead_manager: Optional[BulkheadManager] = None


def get_bulkhead_manager() -> BulkheadManager:
    """Get global bulkhead manager instance."""
    global _bulkhead_manager
    if _bulkhead_manager is None:
        _bulkhead_manager = BulkheadManager()
    return _bulkhead_manager
