"""
QBITEL Engine - Model Training Infrastructure

This module provides comprehensive model training infrastructure including
MLflow integration, distributed training, and automated hyperparameter tuning.
"""

from .trainer import ModelTrainer, TrainingJob, TrainingConfig

# Import optional modules only if they exist
try:
    from .datasets import DatasetManager, ProtocolDataset, AnomalyDataset
except ImportError:
    DatasetManager = None
    ProtocolDataset = None
    AnomalyDataset = None

try:
    from .optimizers import OptimizerFactory, CustomOptimizers
except ImportError:
    OptimizerFactory = None
    CustomOptimizers = None

try:
    from .schedulers import SchedulerFactory, CustomSchedulers
except ImportError:
    SchedulerFactory = None
    CustomSchedulers = None

__all__ = [
    "ModelTrainer",
    "TrainingJob",
    "TrainingConfig",
    "DatasetManager",
    "ProtocolDataset",
    "AnomalyDataset",
    "OptimizerFactory",
    "CustomOptimizers",
    "SchedulerFactory",
    "CustomSchedulers",
]
