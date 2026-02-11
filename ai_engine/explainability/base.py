"""
QBITEL - Base Explainability Interfaces

Defines abstract base classes and common data structures for AI explainability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np


class ExplanationType(str, Enum):
    """Types of explanations supported."""

    SHAP = "shap"  # Shapley Additive Explanations (deep learning)
    LIME = "lime"  # Local Interpretable Model-agnostic Explanations
    ATTENTION = "attention"  # Attention weights (transformers)
    GRADIENT = "gradient"  # Gradient-based attribution
    RULE_BASED = "rule_based"  # Explicit rule explanations


class FeatureImportanceMethod(str, Enum):
    """Methods for calculating feature importance."""

    ABSOLUTE = "absolute"  # Absolute importance values
    RELATIVE = "relative"  # Relative to max importance
    NORMALIZED = "normalized"  # Normalized to sum to 1.0


@dataclass
class FeatureImportance:
    """Feature importance information."""

    feature_name: str
    feature_value: Any
    importance_score: float
    rank: int
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationResult:
    """
    Unified explanation result structure.

    This is returned by all explainer implementations to provide
    consistent explanation format across different methods.
    """

    # Identification
    explanation_id: str
    decision_id: str  # Links to the decision being explained
    timestamp: datetime

    # Model information
    model_name: str
    model_version: str
    explanation_method: ExplanationType

    # Input/Output
    input_data: Any  # Original input to the model
    model_output: Any  # Model's prediction/decision
    confidence_score: float  # Model confidence (0-1)

    # Explanation details
    feature_importances: List[FeatureImportance]
    top_features: List[FeatureImportance]  # Top N most important
    explanation_summary: str  # Human-readable summary

    # Additional context
    counterfactual: Optional[str] = None  # "What if..." explanation
    regulatory_justification: Optional[str] = None  # Compliance context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "explanation_id": self.explanation_id,
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "explanation_method": self.explanation_method.value,
            "input_data": str(self.input_data)[:1000],  # Truncate for safety
            "model_output": str(self.model_output),
            "confidence_score": self.confidence_score,
            "feature_importances": [
                {
                    "feature_name": fi.feature_name,
                    "feature_value": str(fi.feature_value),
                    "importance_score": fi.importance_score,
                    "rank": fi.rank,
                    "description": fi.description,
                }
                for fi in self.feature_importances
            ],
            "top_features": [
                {
                    "feature_name": fi.feature_name,
                    "feature_value": str(fi.feature_value),
                    "importance_score": fi.importance_score,
                    "rank": fi.rank,
                }
                for fi in self.top_features
            ],
            "explanation_summary": self.explanation_summary,
            "counterfactual": self.counterfactual,
            "regulatory_justification": self.regulatory_justification,
            "metadata": self.metadata,
        }


class BaseExplainer(ABC):
    """
    Abstract base class for all explainers.

    All explainer implementations (SHAP, LIME, etc.) must inherit from
    this class and implement the explain() method.
    """

    def __init__(self, model: Any, model_name: str, model_version: str):
        """
        Initialize explainer.

        Args:
            model: The model to explain (PyTorch model, LLM, etc.)
            model_name: Human-readable model name
            model_version: Model version for audit trail
        """
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.explanation_method = self._get_explanation_method()

    @abstractmethod
    def _get_explanation_method(self) -> ExplanationType:
        """Return the explanation method type."""
        pass

    @abstractmethod
    def explain(
        self,
        input_data: Any,
        decision_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> ExplanationResult:
        """
        Generate explanation for a model decision.

        Args:
            input_data: Input data that was fed to the model
            decision_id: Unique identifier for this decision (for audit trail)
            top_k: Number of top features to include in explanation
            **kwargs: Additional explainer-specific parameters

        Returns:
            ExplanationResult with feature importances and summary
        """
        pass

    @abstractmethod
    def batch_explain(
        self,
        input_batch: List[Any],
        decision_ids: List[str],
        top_k: int = 10,
    ) -> List[ExplanationResult]:
        """
        Generate explanations for a batch of decisions.

        Args:
            input_batch: List of input data
            decision_ids: List of decision IDs (same length as input_batch)
            top_k: Number of top features per explanation

        Returns:
            List of ExplanationResult objects
        """
        pass

    def _generate_summary(
        self,
        top_features: List[FeatureImportance],
        model_output: Any,
        confidence: float,
    ) -> str:
        """
        Generate human-readable explanation summary.

        Args:
            top_features: Top important features
            model_output: Model's prediction
            confidence: Confidence score

        Returns:
            Human-readable summary string
        """
        if not top_features:
            return f"Model predicted {model_output} with {confidence:.1%} confidence (no feature importance available)"

        top_feature_names = [f.feature_name for f in top_features[:3]]
        feature_list = ", ".join(top_feature_names)

        return (
            f"Model predicted {model_output} with {confidence:.1%} confidence. "
            f"Key factors: {feature_list}."
        )

    def _rank_features(
        self,
        feature_importances: List[FeatureImportance],
        method: FeatureImportanceMethod = FeatureImportanceMethod.ABSOLUTE,
    ) -> List[FeatureImportance]:
        """
        Rank features by importance.

        Args:
            feature_importances: List of feature importances
            method: Ranking method

        Returns:
            Sorted list with rank field populated
        """
        # Sort by absolute importance (descending)
        sorted_features = sorted(
            feature_importances,
            key=lambda x: abs(x.importance_score),
            reverse=True,
        )

        # Assign ranks
        for rank, feature in enumerate(sorted_features, start=1):
            feature.rank = rank

        return sorted_features


class ExplainerRegistry:
    """
    Registry for managing multiple explainer instances.

    Allows looking up appropriate explainer for a given model.
    """

    def __init__(self):
        self._explainers: Dict[str, BaseExplainer] = {}

    def register(self, model_name: str, explainer: BaseExplainer):
        """Register an explainer for a model."""
        self._explainers[model_name] = explainer

    def get(self, model_name: str) -> Optional[BaseExplainer]:
        """Get explainer for a model."""
        return self._explainers.get(model_name)

    def list_models(self) -> List[str]:
        """List all models with registered explainers."""
        return list(self._explainers.keys())
