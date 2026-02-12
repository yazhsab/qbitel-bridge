"""
QBITEL Engine - Protocol Discovery Module

This module provides AI-powered protocol discovery capabilities including
PCFG grammar inference, statistical analysis, and protocol classification.

Features:
- Protocol classification using Transformer/CNN/LSTM ensemble
- PCFG grammar inference for protocol structure learning
- Statistical analysis of byte patterns and distributions
- Pattern extraction for protocol reverse engineering
- Few-shot learning for rare protocol identification
- Attention-based explainability for classification decisions
"""

from .pcfg_inference import PCFGInference, Grammar, ProductionRule
from .protocol_classifier import ProtocolClassifier, ClassificationResult
from .statistical_analyzer import StatisticalAnalyzer, ByteFrequency, FieldStatistics
from .pattern_extractor import PatternExtractor, PatternResult

# Transformer-based classifier (new architecture)
from .transformer_classifier import (
    ProtocolTransformerClassifier,
    ProtocolTransformer,
    TransformerConfig,
    ClassificationResult as TransformerClassificationResult,
    ProtocolSample,
    create_transformer_classifier,
)

# Hybrid classifier combining Transformer + Legacy
from .hybrid_classifier import (
    HybridProtocolClassifier,
    HybridClassifierConfig,
    HybridClassificationResult,
    create_hybrid_classifier,
)

# Protocol signature database
from .protocol_signatures import (
    ProtocolSignatureDatabase,
    ProtocolSignature,
    SignatureMatch,
    ProtocolCategory,
    EncodingType,
    FramingType,
    get_signature_database,
)

__all__ = [
    # Legacy components
    "PCFGInference",
    "Grammar",
    "ProductionRule",
    "ProtocolClassifier",
    "ClassificationResult",
    "StatisticalAnalyzer",
    "ByteFrequency",
    "FieldStatistics",
    "PatternExtractor",
    "PatternResult",
    # Transformer classifier (new)
    "ProtocolTransformerClassifier",
    "ProtocolTransformer",
    "TransformerConfig",
    "TransformerClassificationResult",
    "ProtocolSample",
    "create_transformer_classifier",
    # Hybrid classifier (recommended)
    "HybridProtocolClassifier",
    "HybridClassifierConfig",
    "HybridClassificationResult",
    "create_hybrid_classifier",
    # Protocol signatures
    "ProtocolSignatureDatabase",
    "ProtocolSignature",
    "SignatureMatch",
    "ProtocolCategory",
    "EncodingType",
    "FramingType",
    "get_signature_database",
]
