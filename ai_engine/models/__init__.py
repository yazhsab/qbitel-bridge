"""Top-level exports for the CRONOS AI model toolkit.

This module makes the commonly used model primitives available from
``ai_engine.models`` so internal and integration code can rely on the
package without importing private modules. The previous implementation
referenced non-existent modules (for example ``base_model`` and
``transformer_models``) which caused an immediate ``ModuleNotFoundError``
whenever ``ai_engine.models`` was imported.  The exports below map to the
actual module structure that exists in the repository today.
"""

from .base import BaseModel, ModelInput, ModelOutput, ModelState
from .ensemble import (
    EnsembleModel,
    EnsembleMethod,
    VotingType,
    EnsembleMember,
    EnsembleResult,
)
from .model_manager import ModelManager
from .registry import (
    ModelRegistry,
    ModelMetadata,
    ModelVersion,
    ModelStatus,
    ModelType,
)

__all__ = [
    # Base model primitives
    "BaseModel",
    "ModelInput",
    "ModelOutput",
    "ModelState",

    # Ensemble tooling
    "EnsembleModel",
    "EnsembleMethod",
    "VotingType",
    "EnsembleMember",
    "EnsembleResult",

    # Management interfaces
    "ModelManager",
    "ModelRegistry",
    "ModelMetadata",
    "ModelVersion",
    "ModelStatus",
    "ModelType",
]
