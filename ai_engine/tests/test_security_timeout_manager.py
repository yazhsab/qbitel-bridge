import asyncio
from typing import Dict

import pytest

from ai_engine.security.resilience.timeout_manager import (
    TimeoutConfig,
    TimeoutContext,
    TimeoutHistory,
    TimeoutManager,
    TimeoutPolicy,
    TimeoutStrategy,
    TimeoutResult,
    get_timeout_manager,
)
from ai_engine.security.logging import SecurityLogType


class _DummyLogger:
    def __init__(self, name: str):
        self.name = name
        self.events = []

    def log_security_event(
        self, log_type, message, level=None, metadata=None, **kwargs
    ):
        entry = {
            "log_type": log_type,
            "message": message,
            "level": level,
            "metadata": metadata or {},
        }
        entry.update(kwargs)
        self.events.append(entry)
        return f"{self.name}:{len(self.events)}"


@pytest.fixture(autouse=True)
def stub_security_logger(monkeypatch) -> Dict[str, _DummyLogger]:
    loggers: Dict[str, _DummyLogger] = {}

    def _get_logger(name: str = "qbitel.security") -> _DummyLogger:
        if name not in loggers:
            loggers[name] = _DummyLogger(name)
        return loggers[name]

    monkeypatch.setattr(
        "ai_engine.security.resilience.timeout_manager.get_security_logger",
        _get_logger,
    )
    return loggers


@pytest.fixture(autouse=True)
def reset_global_timeout_manager():
    import ai_engine.security.resilience.timeout_manager as module

    module._timeout_manager = None
    yield
    module._timeout_manager = None


def test_timeout_history_statistics():
    history = TimeoutHistory(max_size=5)
    history.add_result(1.0, 1.5, True)
    history.add_result(2.0, 1.0, False)
    history.add_result(0.5, 1.0, True)

    assert history.get_average_duration() == pytest.approx(1.1666666, rel=1e-3)
    assert history.get_percentile_duration(50.0) == pytest.approx(1.0, rel=1e-3)
    assert history.get_percentile_duration(95.0) == pytest.approx(2.0, rel=1e-3)
    assert history.get_success_rate() == pytest.approx(2 / 3, rel=1e-3)
    assert history.get_timeout_rate() == pytest.approx(1 / 3, rel=1e-3)


def test_timeout_policy_adaptive_low_success_increases_timeout():
    config = TimeoutConfig(
        name="adaptive",
        default_timeout=10.0,
        strategy=TimeoutStrategy.ADAPTIVE,
        min_timeout=1.0,
        max_timeout=30.0,
        adaptation_factor=1.5,
    )
    policy = TimeoutPolicy(config)
    policy.history.add_result(2.0, 10.0, True)
    policy.history.add_result(4.0, 10.0, False)
    policy.history.add_result(3.0, 10.0, True)

    timeout = policy.calculate_timeout()

    assert timeout == pytest.approx(6.75, rel=1e-3)


def test_timeout_policy_percentile_uses_history():
    config = TimeoutConfig(
        name="percentile",
        default_timeout=5.0,
        strategy=TimeoutStrategy.PERCENTILE,
        min_timeout=1.0,
        max_timeout=10.0,
        percentile=90.0,
    )
    policy = TimeoutPolicy(config)
    for duration in [1.0, 2.0, 3.0, 5.0]:
        policy.history.add_result(duration, 10.0, True)

    timeout = policy.calculate_timeout()

    assert timeout == pytest.approx(6.0, rel=1e-3)


def test_timeout_policy_cascading_increases_with_high_timeout_rate():
    config = TimeoutConfig(
        name="cascade",
        default_timeout=5.0,
        strategy=TimeoutStrategy.CASCADING,
        min_timeout=1.0,
        max_timeout=10.0,
    )
    policy = TimeoutPolicy(config)
    policy.history.add_result(0.5, 1.0, True)
    policy.history.add_result(2.0, 1.0, False)
    policy.history.add_result(3.0, 1.0, False)

    timeout = policy.calculate_timeout()

    assert timeout == pytest.approx(9.625, rel=1e-3)


@pytest.mark.asyncio
async def test_execute_with_timeout_success_updates_metrics():
    config = TimeoutConfig(
        name="runner",
        default_timeout=0.2,
        strategy=TimeoutStrategy.FIXED,
        min_timeout=0.01,
        max_timeout=1.0,
    )
    policy = TimeoutPolicy(config)

    async def short_task(value):
        await asyncio.sleep(0.01)
        return value

    result: TimeoutResult = await policy.execute_with_timeout(short_task, "done")

    assert result.success is True
    assert result.timeout_occurred is False
    metrics = policy.get_metrics()
    assert metrics["statistics"]["total_operations"] == 1
    assert metrics["statistics"]["success_rate"] == pytest.approx(1.0)

    policy.reset_history()
    assert policy.get_metrics()["history_metrics"]["history_size"] == 0


@pytest.mark.asyncio
async def test_execute_with_timeout_reports_timeout():
    config = TimeoutConfig(
        name="timeout",
        default_timeout=0.05,
        strategy=TimeoutStrategy.FIXED,
        min_timeout=0.01,
        max_timeout=0.1,
    )
    policy = TimeoutPolicy(config)

    async def slow_task():
        await asyncio.sleep(0.2)

    result = await policy.execute_with_timeout(slow_task)

    assert result.success is False
    assert result.timeout_occurred is True
    assert "timed out" in (result.error_message or "")
    assert policy.get_metrics()["statistics"]["timeout_operations"] == 1


@pytest.mark.asyncio
async def test_execute_with_timeout_handles_exception():
    config = TimeoutConfig(
        name="failure",
        default_timeout=0.2,
        strategy=TimeoutStrategy.FIXED,
    )
    policy = TimeoutPolicy(config)

    async def failing_task():
        raise RuntimeError("boom")

    result = await policy.execute_with_timeout(failing_task)

    assert result.success is False
    assert result.timeout_occurred is False
    assert result.error_message == "boom"
    assert policy.get_metrics()["statistics"]["failed_operations"] == 1


@pytest.mark.asyncio
async def test_timeout_manager_execute_with_unknown_policy_falls_back(
    stub_security_logger,
):
    manager = TimeoutManager()

    async def quick():
        await asyncio.sleep(0)
        return "ok"

    result = await manager.execute_with_policy("nonexistent", quick)

    assert result.success is True
    manager_logger = stub_security_logger["qbitel.security.resilience.timeout_manager"]
    assert any(
        "Using default timeout policy" in event["message"]
        for event in manager_logger.events
    )


@pytest.mark.asyncio
async def test_timeout_manager_global_metrics_combine_policy_stats():
    manager = TimeoutManager()

    policy_name = "test-fast"
    manager.create_policy(
        name=policy_name,
        default_timeout=0.05,
        strategy=TimeoutStrategy.FIXED,
        min_timeout=0.01,
        max_timeout=0.5,
    )

    async def succeed():
        await asyncio.sleep(0.0)

    def failing_sync():
        raise ValueError("failure")

    async def slow():
        await asyncio.sleep(0.2)

    success_result = await manager.execute_with_policy(policy_name, succeed)
    failure_result = await manager.execute_with_policy(policy_name, failing_sync)
    timeout_result = await manager.execute_with_policy(policy_name, slow)

    global_metrics = manager.get_global_metrics()
    stats = global_metrics["global_statistics"]
    fast_metrics = global_metrics["policies"][policy_name]["statistics"]

    assert success_result.success is True
    assert failure_result.success is False
    assert timeout_result.timeout_occurred is True
    assert stats["total_operations"] == 3
    assert fast_metrics["successful_operations"] == 1
    assert fast_metrics["failed_operations"] == 2
    assert fast_metrics["timeout_operations"] == 1


@pytest.mark.asyncio
async def test_timeout_context_logs_timeout_event(stub_security_logger):
    async def slow_operation():
        manager = TimeoutManager()
        async with TimeoutContext(
            manager,
            policy_name="fast",
            operation_name="slow-op",
        ):
            raise asyncio.TimeoutError()

    with pytest.raises(asyncio.TimeoutError):
        await slow_operation()

    context_logger = stub_security_logger["qbitel.security.resilience.timeout_context"]
    assert any(
        event["log_type"] == SecurityLogType.PERFORMANCE_METRIC
        and "slow-op" in event["message"]
        for event in context_logger.events
    )


def test_get_timeout_manager_returns_singleton():
    manager_one = get_timeout_manager()
    manager_two = get_timeout_manager()

    assert manager_one is manager_two
