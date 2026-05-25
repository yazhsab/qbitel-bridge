"""
Workforce Management Integration Bridges

Provides quantum-safe bridges to Workforce Management systems for
agent scheduling, real-time adherence, forecasting, and performance
metrics with integrated QBITEL session management.
"""

from typing import List

__all__: List[str] = [
    "WFMBridge",
    "WFMType",
    "WFMConnectionConfig",
    "AgentSchedule",
    "PerformanceMetrics",
    "ForecastData",
]


def __getattr__(name: str):
    """Lazy import WFM bridge classes."""
    if name in ("WFMBridge", "WFMType", "WFMConnectionConfig",
                "AgentSchedule", "PerformanceMetrics", "ForecastData"):
        from ai_engine.domains.bpo.integrations.wfm.wfm_bridge import (
            WFMBridge,
            WFMType,
            WFMConnectionConfig,
            AgentSchedule,
            PerformanceMetrics,
            ForecastData,
        )
        _map = {
            "WFMBridge": WFMBridge,
            "WFMType": WFMType,
            "WFMConnectionConfig": WFMConnectionConfig,
            "AgentSchedule": AgentSchedule,
            "PerformanceMetrics": PerformanceMetrics,
            "ForecastData": ForecastData,
        }
        return _map[name]

    raise AttributeError(f"module 'ai_engine.domains.bpo.integrations.wfm' has no attribute '{name}'")
