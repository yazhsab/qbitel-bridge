import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

from ai_engine.security.monitoring import (
    HealthCheckManager,
    HealthStatus,
    PerformanceMetrics,
    PerformanceMonitor,
    PrometheusMetrics,
    SecurityOrchestrationMonitor,
)


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


class _StubProcess:
    def __init__(self, _pid=None):
        self._cpu_percent_sequence = [12.5, 17.0]

    def cpu_percent(self):
        # Rotate through sequence to simulate changing CPU load
        value = self._cpu_percent_sequence.pop(0)
        self._cpu_percent_sequence.append(value)
        return value

    def memory_info(self):
        return SimpleNamespace(rss=12 * 1024 * 1024)

    def memory_percent(self):
        return 42.0


@pytest.fixture(autouse=True)
def stub_security_config(monkeypatch):
    class MonitoringSettings:
        def __init__(self):
            self.health_checks = {"interval_seconds": 0.01, "timeout_seconds": 0.05}

    class StubConfig:
        def __init__(self):
            self.environment = "test"
            self.monitoring = MonitoringSettings()

    monkeypatch.setattr(
        "ai_engine.security.monitoring.get_security_config",
        lambda: StubConfig(),
    )


@pytest.fixture(autouse=True)
def stub_security_logger(monkeypatch) -> Dict[str, _DummyLogger]:
    loggers: Dict[str, _DummyLogger] = {}

    def _get_logger(name: str = "cronos.security") -> _DummyLogger:
        if name not in loggers:
            loggers[name] = _DummyLogger(name)
        return loggers[name]

    monkeypatch.setattr(
        "ai_engine.security.monitoring.get_security_logger",
        _get_logger,
    )
    return loggers


@pytest.fixture(autouse=True)
def stub_psutil(monkeypatch):
    monkeypatch.setattr("ai_engine.security.monitoring.psutil.Process", _StubProcess)


@pytest.mark.asyncio
async def test_health_check_manager_runs_registered_checks(stub_security_logger):
    manager = HealthCheckManager()

    manager.register_health_check(
        "database",
        lambda: {
            "status": HealthStatus.HEALTHY,
            "message": "OK",
            "details": {"latency_ms": 12},
        },
    )

    manager.register_health_check("cache", lambda: False)

    result_db = await manager.run_health_check("database")
    result_cache = await manager.run_health_check("cache")

    assert result_db.status is HealthStatus.HEALTHY
    assert result_db.details["latency_ms"] == 12
    assert result_cache.status is HealthStatus.UNHEALTHY

    overall = manager.get_overall_health()
    assert overall is HealthStatus.UNHEALTHY

    unknown = await manager.run_health_check("missing")
    assert unknown.status is HealthStatus.UNKNOWN

    # Verify logging captured the events
    logger = stub_security_logger["cronos.security.monitoring"]
    assert any(
        "Health check for database" in event["message"] for event in logger.events
    )


@pytest.mark.asyncio
async def test_health_check_manager_background_loop_updates_results():
    manager = HealthCheckManager()
    manager.register_health_check("api", lambda: True)

    await manager.start_background_checks()
    # Allow loop to execute once
    await asyncio.sleep(0.02)
    await manager.stop_background_checks()

    assert "api" in manager.health_results
    assert manager.health_results["api"].status is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_health_check_manager_handles_exceptions():
    manager = HealthCheckManager()
    manager.register_health_check(
        "explosive", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    result = await manager.run_health_check("explosive")

    assert result.status is HealthStatus.UNHEALTHY
    assert "boom" in result.message


def test_performance_monitor_tracks_metrics(stub_security_logger):
    prometheus_metrics = PrometheusMetrics()
    monitor = PerformanceMonitor(prometheus_metrics)

    monitor.record_security_event(50.0)
    monitor.record_security_event(100.0, success=False)
    monitor.record_decision()
    monitor.record_decision(auto_executed=True)

    metrics = monitor.get_current_metrics()
    assert metrics.error_rate == pytest.approx(0.5)
    assert metrics.events_per_second >= 0.0

    monitor.update_prometheus_metrics(metrics)

    logger = stub_security_logger["cronos.security.monitoring"]
    monitor_logger_messages = [event["message"] for event in logger.events]
    assert (
        any("Performance monitoring started" in msg for msg in monitor_logger_messages)
        is False
    )


@pytest.mark.asyncio
async def test_performance_monitor_start_stop_logs_events(stub_security_logger):
    prometheus_metrics = PrometheusMetrics()
    monitor = PerformanceMonitor(prometheus_metrics)

    await monitor.start_monitoring()
    await asyncio.sleep(0)
    await monitor.stop_monitoring()

    logger = stub_security_logger["cronos.security.monitoring"]
    messages = [event["message"] for event in logger.events]
    assert any("Performance monitoring started" in msg for msg in messages)
    assert any("Performance monitoring stopped" in msg for msg in messages)


@pytest.mark.asyncio
async def test_security_monitor_lifecycle(monkeypatch, stub_security_logger):
    monitor = SecurityOrchestrationMonitor()

    captured_info: Dict[str, Dict[str, Any]] = {}

    def _capture_info(payload: Dict[str, Any]):
        captured_info["data"] = payload

    monkeypatch.setattr(monitor.prometheus_metrics.service_info, "info", _capture_info)
    monkeypatch.setattr(
        monitor,
        "_register_default_health_checks",
        AsyncMock(),
    )

    monitor.health_check_manager.start_background_checks = AsyncMock()
    monitor.health_check_manager.stop_background_checks = AsyncMock()
    monitor.performance_monitor.start_monitoring = AsyncMock()
    monitor.performance_monitor.stop_monitoring = AsyncMock()
    monitor.performance_monitor.record_security_event = (
        lambda duration, success=True: None
    )

    await monitor.initialize()
    assert monitor._initialized is True
    assert captured_info["data"]["environment"] == "test"

    await monitor.start()
    assert monitor._running is True
    monitor.record_security_event_processing(12.5)
    monitor.record_decision_made(auto_executed=True)

    await monitor.stop()
    assert monitor._running is False

    logger = stub_security_logger["cronos.security.monitoring"]
    messages = [event["message"] for event in logger.events]
    assert any("Security orchestration monitor initialized" in msg for msg in messages)
    assert any(
        "Security orchestration monitor started successfully" in msg for msg in messages
    )
    assert any(
        "Security orchestration monitor stopped successfully" in msg for msg in messages
    )


def test_security_monitor_reporting(monkeypatch):
    monitor = SecurityOrchestrationMonitor()

    monitor.health_check_manager.health_results = {
        "api": SimpleNamespace(
            to_dict=lambda: {"status": HealthStatus.HEALTHY.value},
            status=HealthStatus.HEALTHY,
        )
    }

    sample_metrics = PerformanceMetrics(
        timestamp=datetime.now(tz=UTC),
        cpu_percent=10.0,
        memory_percent=20.0,
        memory_used_mb=30.0,
        active_threads=4,
        active_incidents=1,
        active_quarantines=2,
        events_per_second=3.0,
        decisions_per_minute=4.0,
        average_response_time_ms=5.0,
        error_rate=0.1,
    )

    monkeypatch.setattr(
        monitor.performance_monitor,
        "get_current_metrics",
        lambda: sample_metrics,
    )

    health = monitor.get_health_status()
    performance = monitor.get_performance_metrics()
    prometheus = monitor.get_prometheus_metrics()

    assert health["overall_status"] == HealthStatus.HEALTHY.value
    assert performance["cpu_percent"] == 10.0
    assert isinstance(prometheus, str)
