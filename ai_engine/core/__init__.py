"""
CRONOS AI Engine - Core Module

This module provides the core infrastructure for the AI Engine including
configuration management, exception handling, and the main engine orchestrator.
"""

from .engine import CronosAIEngine
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