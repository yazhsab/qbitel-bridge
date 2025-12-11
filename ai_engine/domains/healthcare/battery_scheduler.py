"""
Battery-Aware Cryptographic Scheduler

Schedules cryptographic operations based on device power state
to maximize battery life in implantable medical devices.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PowerProfile(Enum):
    """Device power profiles."""

    CRITICAL = auto()    # <10% battery, defer all non-essential crypto
    LOW = auto()         # 10-25% battery, minimize operations
    NORMAL = auto()      # 25-75% battery, standard operations
    HIGH = auto()        # >75% battery, can perform expensive operations
    CHARGING = auto()    # Device is charging (if applicable)


@dataclass
class CryptoOperation:
    """A cryptographic operation to be scheduled."""

    operation_id: str
    operation_type: str  # "keygen", "sign", "verify", "encapsulate", "decrypt"
    priority: int  # 1 (highest) to 10 (lowest)
    estimated_power_mw: float
    deadline: Optional[float] = None  # Unix timestamp
    callback: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline


class BatteryAwareCryptoScheduler:
    """
    Scheduler that defers expensive cryptographic operations
    based on battery state.

    Strategies:
    1. Queue non-urgent operations when battery is low
    2. Batch operations during high-power periods
    3. Use pre-computed keys during critical battery
    4. Emergency bypass for life-critical operations
    """

    def __init__(
        self,
        device_id: str,
        initial_power_profile: PowerProfile = PowerProfile.NORMAL,
    ):
        self.device_id = device_id
        self._power_profile = initial_power_profile
        self._pending_operations: List[CryptoOperation] = []
        self._running = False

        # Power thresholds (mW) per profile
        self._power_budgets = {
            PowerProfile.CRITICAL: 0.5,
            PowerProfile.LOW: 2.0,
            PowerProfile.NORMAL: 10.0,
            PowerProfile.HIGH: 50.0,
            PowerProfile.CHARGING: 100.0,
        }

        logger.info(f"Battery scheduler initialized for {device_id}")

    def update_power_profile(self, profile: PowerProfile) -> None:
        """Update the current power profile."""
        old_profile = self._power_profile
        self._power_profile = profile

        if profile != old_profile:
            logger.info(f"Power profile changed: {old_profile.name} -> {profile.name}")

            # Process queued operations if power improved
            if self._can_process_queue():
                asyncio.create_task(self._process_queue())

    def _can_process_queue(self) -> bool:
        """Check if we can process queued operations."""
        return self._power_profile in (
            PowerProfile.NORMAL,
            PowerProfile.HIGH,
            PowerProfile.CHARGING,
        )

    async def schedule_operation(
        self,
        operation: CryptoOperation,
    ) -> bool:
        """
        Schedule a cryptographic operation.

        Returns True if operation can be executed immediately,
        False if it was queued for later.
        """
        budget = self._power_budgets[self._power_profile]

        # High priority operations always execute
        if operation.priority <= 2:
            await self._execute_operation(operation)
            return True

        # Check if operation fits power budget
        if operation.estimated_power_mw <= budget:
            await self._execute_operation(operation)
            return True

        # Queue for later
        self._pending_operations.append(operation)
        self._pending_operations.sort(key=lambda x: x.priority)

        logger.debug(
            f"Queued operation {operation.operation_id} "
            f"(power={operation.estimated_power_mw}mW, budget={budget}mW)"
        )

        return False

    async def _execute_operation(self, operation: CryptoOperation) -> None:
        """Execute a cryptographic operation."""
        if operation.callback:
            try:
                if asyncio.iscoroutinefunction(operation.callback):
                    await operation.callback(*operation.args, **operation.kwargs)
                else:
                    operation.callback(*operation.args, **operation.kwargs)
            except Exception as e:
                logger.error(f"Operation {operation.operation_id} failed: {e}")

    async def _process_queue(self) -> None:
        """Process queued operations."""
        if not self._pending_operations:
            return

        budget = self._power_budgets[self._power_profile]
        processed = []

        for op in self._pending_operations:
            if op.is_expired:
                processed.append(op)
                continue

            if op.estimated_power_mw <= budget:
                await self._execute_operation(op)
                processed.append(op)

        for op in processed:
            self._pending_operations.remove(op)

        if processed:
            logger.info(f"Processed {len(processed)} queued operations")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "device_id": self.device_id,
            "power_profile": self._power_profile.name,
            "power_budget_mw": self._power_budgets[self._power_profile],
            "queued_operations": len(self._pending_operations),
            "oldest_operation_age": (
                time.time() - min(
                    (op.deadline or time.time() for op in self._pending_operations),
                    default=time.time()
                )
                if self._pending_operations else 0
            ),
        }
