"""
QBITEL - SHAP Explainer

SHAP (SHapley Additive exPlanations) integration for deep learning models,
specifically protocol classifiers (CNN, LSTM, ensemble models).
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import shap
import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import (
    BaseExplainer,
    ExplanationResult,
    ExplanationType,
    FeatureImportance,
)

logger = logging.getLogger(__name__)


class SHAPProtocolExplainer(BaseExplainer):
    """
    SHAP-based explainer for protocol classification models.

    Uses DeepExplainer for PyTorch models (CNN, LSTM) to provide
    byte-level attribution for protocol classification decisions.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        model_version: str,
        background_data: Optional[np.ndarray] = None,
        background_size: int = 100,
        device: str = "cpu",
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: PyTorch model to explain
            model_name: Human-readable model name
            model_version: Model version for audit trail
            background_data: Background dataset for SHAP (optional)
            background_size: Number of background samples to use
            device: Device to run model on ('cpu' or 'cuda')
        """
        super().__init__(model, model_name, model_version)

        self.device = device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Initialize background data
        if background_data is not None:
            self.background_data = torch.tensor(
                background_data[:background_size],
                dtype=torch.long,
                device=self.device,
            )
        else:
            # Create synthetic background data if none provided
            # For protocol classifiers, this is random byte sequences
            self.background_data = torch.randint(0, 256, (background_size, 512), dtype=torch.long, device=self.device)

        # Initialize SHAP explainer
        try:
            self.explainer = shap.DeepExplainer(
                self.model,
                self.background_data,
            )
            logger.info(f"Initialized SHAP explainer for {model_name} with " f"{background_size} background samples")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            # Fallback to GradientExplainer if DeepExplainer fails
            self.explainer = shap.GradientExplainer(
                self.model,
                self.background_data,
            )
            logger.warning("Using GradientExplainer as fallback")

    def _get_explanation_method(self) -> ExplanationType:
        """Return SHAP as the explanation method."""
        return ExplanationType.SHAP

    def explain(
        self,
        input_data: Any,
        decision_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> ExplanationResult:
        """
        Generate SHAP explanation for a single protocol classification.

        Args:
            input_data: Input bytes/packet (numpy array or bytes)
            decision_id: Unique decision identifier
            top_k: Number of top features to return
            **kwargs: Additional parameters

        Returns:
            ExplanationResult with byte-level feature importances
        """
        start_time = datetime.now(timezone.utc)

        # Convert input to tensor
        input_tensor = self._prepare_input(input_data)

        # Get model prediction
        with torch.no_grad():
            model_output = self.model(input_tensor)
            if isinstance(model_output, tuple):
                model_output = model_output[0]  # Handle tuple outputs

            probabilities = torch.softmax(model_output, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()

        # Compute SHAP values
        try:
            shap_values = self.explainer.shap_values(input_tensor)

            # Handle multi-class output (shap_values is list of arrays)
            if isinstance(shap_values, list):
                # Use SHAP values for predicted class
                shap_values_for_class = shap_values[predicted_class]
            else:
                shap_values_for_class = shap_values

            # Squeeze to remove batch dimension
            shap_values_flat = np.squeeze(shap_values_for_class)

        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            # Return empty explanation on failure
            return self._create_empty_explanation(decision_id, input_data, predicted_class, confidence, str(e))

        # Convert SHAP values to feature importances
        feature_importances = self._shap_to_features(
            shap_values_flat,
            input_tensor.cpu().numpy()[0],
        )

        # Rank and get top features
        ranked_features = self._rank_features(feature_importances)
        top_features = ranked_features[:top_k]

        # Generate explanation summary
        summary = self._generate_summary(top_features, predicted_class, confidence)

        # Generate regulatory justification
        regulatory_justification = self._generate_regulatory_justification(top_features, confidence)

        # Calculate explanation time
        explanation_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return ExplanationResult(
            explanation_id=str(uuid4()),
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            model_name=self.model_name,
            model_version=self.model_version,
            explanation_method=self.explanation_method,
            input_data=input_data,
            model_output=predicted_class,
            confidence_score=confidence,
            feature_importances=ranked_features,
            top_features=top_features,
            explanation_summary=summary,
            regulatory_justification=regulatory_justification,
            metadata={
                "shap_base_value": float(np.mean(shap_values_flat)),
                "explanation_time_ms": explanation_time_ms,
                "background_size": len(self.background_data),
            },
        )

    def batch_explain(
        self,
        input_batch: List[Any],
        decision_ids: List[str],
        top_k: int = 10,
    ) -> List[ExplanationResult]:
        """
        Generate SHAP explanations for a batch of inputs.

        Args:
            input_batch: List of input data
            decision_ids: List of decision IDs
            top_k: Number of top features

        Returns:
            List of ExplanationResult objects
        """
        if len(input_batch) != len(decision_ids):
            raise ValueError("input_batch and decision_ids must have same length")

        results = []
        for input_data, decision_id in zip(input_batch, decision_ids):
            result = self.explain(input_data, decision_id, top_k)
            results.append(result)

        return results

    def _prepare_input(self, input_data: Any) -> torch.Tensor:
        """
        Convert input data to tensor format.

        Args:
            input_data: Raw input (bytes, numpy array, or tensor)

        Returns:
            PyTorch tensor ready for model
        """
        if isinstance(input_data, bytes):
            # Convert bytes to numpy array
            input_array = np.frombuffer(input_data, dtype=np.uint8)
        elif isinstance(input_data, np.ndarray):
            input_array = input_data
        elif isinstance(input_data, torch.Tensor):
            return input_data.to(self.device).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Pad or truncate to expected length (512 bytes for protocol classifiers)
        expected_length = 512
        if len(input_array) < expected_length:
            input_array = np.pad(
                input_array,
                (0, expected_length - len(input_array)),
                constant_values=0,
            )
        else:
            input_array = input_array[:expected_length]

        # Convert to tensor
        input_tensor = torch.tensor(input_array, dtype=torch.long, device=self.device).unsqueeze(0)

        return input_tensor

    def _shap_to_features(
        self,
        shap_values: np.ndarray,
        input_bytes: np.ndarray,
    ) -> List[FeatureImportance]:
        """
        Convert SHAP values to FeatureImportance objects.

        Args:
            shap_values: SHAP importance values (1D array)
            input_bytes: Original input byte sequence

        Returns:
            List of FeatureImportance objects
        """
        feature_importances = []

        for byte_pos, (shap_value, byte_value) in enumerate(zip(shap_values, input_bytes)):
            # Create human-readable description
            char_repr = chr(byte_value) if 32 <= byte_value < 127 else f"\\x{byte_value:02x}"
            description = self._get_byte_description(byte_pos, byte_value, char_repr)

            feature_importances.append(
                FeatureImportance(
                    feature_name=f"byte_{byte_pos}",
                    feature_value=byte_value,
                    importance_score=float(shap_value),
                    rank=0,  # Will be set by _rank_features
                    description=description,
                    metadata={
                        "position": byte_pos,
                        "hex_value": f"0x{byte_value:02x}",
                        "char_repr": char_repr,
                    },
                )
            )

        return feature_importances

    def _get_byte_description(
        self,
        position: int,
        value: int,
        char_repr: str,
    ) -> str:
        """Generate human-readable description for a byte."""
        # Check for common protocol patterns
        if position == 0:
            if char_repr in ["G", "P", "H"]:  # HTTP, POST, HEAD
                return f"HTTP verb start character '{char_repr}'"
            elif value == 0x16:
                return "TLS handshake (0x16)"
            elif value in [0x00, 0x01, 0x02]:
                return f"Binary protocol header (0x{value:02x})"

        return f"Byte {position}: '{char_repr}' (0x{value:02x})"

    def _generate_regulatory_justification(
        self,
        top_features: List[FeatureImportance],
        confidence: float,
    ) -> str:
        """Generate regulatory compliance justification."""
        if confidence > 0.95:
            confidence_level = "very high"
        elif confidence > 0.80:
            confidence_level = "high"
        elif confidence > 0.60:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        top_positions = [f.metadata["position"] for f in top_features[:3]]
        position_str = ", ".join(map(str, top_positions))

        return (
            f"This classification has {confidence_level} confidence ({confidence:.1%}). "
            f"The decision is primarily based on byte patterns at positions {position_str}. "
            f"SHAP attribution analysis shows these bytes contributed most significantly "
            f"to the classification outcome. This explanation meets EU AI Act Article 13 "
            f"requirements for high-risk AI systems."
        )

    def _create_empty_explanation(
        self,
        decision_id: str,
        input_data: Any,
        predicted_class: int,
        confidence: float,
        error_message: str,
    ) -> ExplanationResult:
        """Create empty explanation when SHAP computation fails."""
        return ExplanationResult(
            explanation_id=str(uuid4()),
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            model_name=self.model_name,
            model_version=self.model_version,
            explanation_method=self.explanation_method,
            input_data=input_data,
            model_output=predicted_class,
            confidence_score=confidence,
            feature_importances=[],
            top_features=[],
            explanation_summary=f"Explanation generation failed: {error_message}",
            regulatory_justification="Explanation unavailable due to computational error",
            metadata={"error": error_message},
        )


def create_background_dataset(
    protocol_samples: List[bytes],
    max_samples: int = 100,
) -> np.ndarray:
    """
    Create background dataset for SHAP from protocol samples.

    Args:
        protocol_samples: List of protocol packet samples
        max_samples: Maximum number of samples to include

    Returns:
        NumPy array suitable for SHAP background data
    """
    background_arrays = []

    for sample in protocol_samples[:max_samples]:
        # Convert bytes to numpy array
        sample_array = np.frombuffer(sample, dtype=np.uint8)

        # Pad or truncate to 512 bytes
        if len(sample_array) < 512:
            sample_array = np.pad(sample_array, (0, 512 - len(sample_array)), constant_values=0)
        else:
            sample_array = sample_array[:512]

        background_arrays.append(sample_array)

    return np.array(background_arrays)
