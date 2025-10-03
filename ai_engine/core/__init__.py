"""
CRONOS AI Engine - Core Module

This module provides the core infrastructure for the AI Engine including
configuration management, exception handling, and the main engine orchestrator.
"""

try:  # pragma: no cover - optional dependency shim for lightweight contexts
    from .engine import CronosAIEngine  # type: ignore
except ModuleNotFoundError:  # torch or other heavy deps may be absent in lightweight test runs
    CronosAIEngine = None  # type: ignore

from .config import Config, ModelConfig, TrainingConfig
from .exceptions import (
    CronosAIException,
    ModelException,
    InferenceException,
    TrainingException,
    ConfigurationException,
)

__version__ = "1.0.0"
__all__ = [
    "CronosAIEngine",
    "Config",
    "ModelConfig", 
    "TrainingConfig",
    "CronosAIException",
    "ModelException",
    "InferenceException",
    "TrainingException",
    "ConfigurationException",
]
