"""
CRONOS AI Engine - Field Detection Module

This module provides AI-powered field detection capabilities using BiLSTM-CRF
models for sequence labeling and boundary detection in protocol messages.
"""

__all__ = []

try:
    from .field_detector import FieldDetector, FieldPrediction, FieldBoundary
except ImportError:  # pragma: no cover - optional modules
    FieldDetector = None  # type: ignore
    FieldPrediction = None  # type: ignore
    FieldBoundary = None  # type: ignore
else:
    __all__.extend(["FieldDetector", "FieldPrediction", "FieldBoundary"])

try:
    from .boundary_detection import BoundaryDetector, BoundaryResult
except ImportError:  # pragma: no cover - optional modules
    BoundaryDetector = None  # type: ignore
    BoundaryResult = None  # type: ignore
else:
    __all__.extend(["BoundaryDetector", "BoundaryResult"])

try:
    from .type_inference import TypeInference, FieldType
except ImportError:  # pragma: no cover - optional modules
    TypeInference = None  # type: ignore
    FieldType = None  # type: ignore
else:
    __all__.extend(["TypeInference", "FieldType"])

try:
    from .sequence_labeling import SequenceLabelingModel, IOBTag
except ImportError:  # pragma: no cover - optional modules
    SequenceLabelingModel = None  # type: ignore
    IOBTag = None  # type: ignore
else:
    __all__.extend(["SequenceLabelingModel", "IOBTag"])
