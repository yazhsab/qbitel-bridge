"""
CRONOS AI - Explainability Integration Example

Example of integrating explainability into the protocol classifier.
This demonstrates how to use SHAP explainer with the existing CNN classifier.
"""

import asyncio
import logging
from typing import Optional, Tuple

import numpy as np
import torch

from .shap_explainer import SHAPProtocolExplainer, create_background_dataset
from .audit_logger import get_audit_logger
from .base import ExplanationResult
from ..discovery.protocol_classifier import ProtocolClassifier
from ..core.config import get_config

logger = logging.getLogger(__name__)


class ExplainableProtocolClassifier:
    """
    Protocol classifier with integrated explainability.

    Extends ProtocolClassifier to provide explanations for classifications.
    """

    def __init__(
        self,
        classifier: ProtocolClassifier,
        database_url: str,
        enable_audit_logging: bool = True,
        enable_explanations: bool = True,
    ):
        """
        Initialize explainable classifier.

        Args:
            classifier: Existing ProtocolClassifier instance
            database_url: Database URL for audit logging
            enable_audit_logging: Whether to log decisions to audit trail
            enable_explanations: Whether to generate explanations
        """
        self.classifier = classifier
        self.enable_audit_logging = enable_audit_logging
        self.enable_explanations = enable_explanations

        # Initialize explainer if enabled
        self.explainer: Optional[SHAPProtocolExplainer] = None
        if enable_explanations:
            self._initialize_explainer()

        # Initialize audit logger if enabled
        self.audit_logger = None
        if enable_audit_logging:
            self.audit_logger = get_audit_logger(database_url)

        logger.info(
            f"Initialized ExplainableProtocolClassifier "
            f"(audit_logging={enable_audit_logging}, explanations={enable_explanations})"
        )

    def _initialize_explainer(self):
        """Initialize SHAP explainer with background data."""
        # Get background data from classifier's training set
        # In production, this would use actual training data
        background_samples = self._get_background_samples()

        if background_samples:
            background_data = create_background_dataset(
                background_samples,
                max_samples=100,
            )

            self.explainer = SHAPProtocolExplainer(
                model=self.classifier.model,
                model_name="protocol_classifier",
                model_version=self.classifier.model_version or "2.1.0",
                background_data=background_data,
                background_size=100,
            )

            logger.info("Initialized SHAP explainer for protocol classifier")
        else:
            logger.warning("No background data available, explainer disabled")

    def _get_background_samples(self) -> list:
        """Get background samples for SHAP explainer."""
        # TODO: In production, fetch from training data storage
        # For now, return empty list (will use synthetic background)
        return []

    async def classify_with_explanation(
        self,
        packet_data: bytes,
        decision_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        compliance_framework: Optional[str] = None,
    ) -> Tuple[str, Optional[ExplanationResult]]:
        """
        Classify packet and optionally generate explanation.

        Args:
            packet_data: Raw packet bytes
            decision_id: Unique decision identifier
            user_id: User triggering classification
            session_id: Session identifier
            compliance_framework: Applicable compliance framework

        Returns:
            Tuple of (protocol_type, explanation_result)
        """
        # Perform classification
        classification_result = self.classifier.classify(packet_data)
        protocol_type = classification_result.protocol_type

        # Generate explanation if enabled
        explanation = None
        if self.enable_explanations and self.explainer:
            try:
                explanation = self.explainer.explain(
                    input_data=packet_data,
                    decision_id=decision_id or f"cls-{id(packet_data)}",
                    top_k=10,
                )
                logger.debug(
                    f"Generated explanation for {protocol_type} classification "
                    f"(confidence={explanation.confidence_score:.2%})"
                )
            except Exception as e:
                logger.error(f"Failed to generate explanation: {e}")

        # Log to audit trail if enabled
        if self.enable_audit_logging and self.audit_logger and explanation:
            try:
                await self.audit_logger.log_decision(
                    explanation=explanation,
                    event_type="protocol_classification",
                    event_data={
                        "packet_size": len(packet_data),
                        "packet_preview": packet_data[:64].hex(),
                    },
                    compliance_framework=compliance_framework,
                    user_id=user_id,
                    session_id=session_id,
                )
            except Exception as e:
                logger.error(f"Failed to log decision to audit trail: {e}")

        return protocol_type, explanation

    def classify_batch_with_explanations(
        self,
        packets: list,
        **kwargs,
    ) -> list:
        """
        Classify batch of packets with explanations.

        Args:
            packets: List of packet bytes
            **kwargs: Additional parameters

        Returns:
            List of (protocol_type, explanation) tuples
        """
        results = []
        for packet in packets:
            result = asyncio.run(self.classify_with_explanation(packet, **kwargs))
            results.append(result)
        return results


# Example usage function
async def example_usage():
    """
    Example of how to use explainable protocol classifier.
    """
    from ..discovery.protocol_classifier import ProtocolClassifier
    from ..core.config import get_config

    # Initialize base classifier
    config = get_config()
    classifier = ProtocolClassifier(config)

    # Wrap with explainability
    explainable_classifier = ExplainableProtocolClassifier(
        classifier=classifier,
        database_url="postgresql+asyncpg://user:pass@localhost/cronos_ai",
        enable_audit_logging=True,
        enable_explanations=True,
    )

    # Classify packet with explanation
    packet = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
    protocol, explanation = await explainable_classifier.classify_with_explanation(
        packet_data=packet,
        decision_id="example-001",
        user_id="analyst-123",
        compliance_framework="SOC2",
    )

    print(f"Classified as: {protocol}")
    if explanation:
        print(f"Confidence: {explanation.confidence_score:.2%}")
        print(f"Summary: {explanation.explanation_summary}")
        print("\nTop Features:")
        for feature in explanation.top_features[:5]:
            print(f"  - {feature.description}: {feature.importance_score:.3f}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
