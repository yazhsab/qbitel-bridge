"""
QBITEL Engine - Core Module

This module provides the core infrastructure for the AI Engine including
configuration management, exception handling, and the main engine orchestrator.
"""

try:  # pragma: no cover - optional dependency shim for lightweight contexts
    from .engine import QbitelAIEngine  # type: ignore
except ModuleNotFoundError:  # torch or other heavy deps may be absent in lightweight test runs
    QbitelAIEngine = None  # type: ignore

from .config import Config, ModelConfig, TrainingConfig
from .exceptions import (
    QbitelAIException,
    ModelException,
    InferenceException,
    TrainingException,
    ConfigurationException,
)

__version__ = "1.0.0"
__all__ = [
    "QbitelAIEngine",
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "QbitelAIException",
    "ModelException",
    "InferenceException",
    "TrainingException",
    "ConfigurationException",
]
