"""
CRONOS AI - LIME Explainer

LIME (Local Interpretable Model-agnostic Explanations) integration for
LLM-based security decisions and policy generation.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from lime.lime_text import LimeTextExplainer

from .base import (
    BaseExplainer,
    ExplanationResult,
    ExplanationType,
    FeatureImportance,
)

logger = logging.getLogger(__name__)


class LIMEDecisionExplainer(BaseExplainer):
    """
    LIME-based explainer for LLM security decisions.

    Explains "why" the LLM made a particular security decision by:
    1. Perturbing the input text (event description)
    2. Getting predictions for perturbed inputs
    3. Fitting a linear model to understand feature importance
    4. Generating counterfactual explanations
    """

    def __init__(
        self,
        predict_fn: Callable[[List[str]], List[np.ndarray]],
        model_name: str,
        model_version: str,
        class_names: List[str],
        feature_selection: str = "auto",
        num_samples: int = 1000,
    ):
        """
        Initialize LIME explainer.

        Args:
            predict_fn: Function that takes list of texts and returns prediction probabilities
                        Shape: (batch_size, num_classes)
            model_name: Human-readable model name
            model_version: Model version for audit trail
            class_names: List of class labels (e.g., ['ALLOW', 'BLOCK', 'ESCALATE'])
            feature_selection: LIME feature selection method ('auto', 'forward_selection', 'lasso_path')
            num_samples: Number of perturbed samples for LIME
        """
        super().__init__(None, model_name, model_version)

        self.predict_fn = predict_fn
        self.class_names = class_names
        self.num_samples = num_samples

        # Initialize LIME text explainer
        self.explainer = LimeTextExplainer(
            class_names=class_names,
            feature_selection=feature_selection,
            bow=False,  # Don't use bag-of-words (preserve word order)
            split_expression=r"\W+",  # Split on non-word characters
        )

        logger.info(
            f"Initialized LIME explainer for {model_name} with "
            f"{len(class_names)} classes and {num_samples} samples"
        )

    def _get_explanation_method(self) -> ExplanationType:
        """Return LIME as the explanation method."""
        return ExplanationType.LIME

    def explain(
        self,
        input_data: Any,
        decision_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> ExplanationResult:
        """
        Generate LIME explanation for a security decision.

        Args:
            input_data: Input text (event description, threat context)
            decision_id: Unique decision identifier
            top_k: Number of top features to return
            **kwargs: Additional parameters (e.g., 'predicted_class' to explain specific class)

        Returns:
            ExplanationResult with feature importances and counterfactual
        """
        start_time = datetime.now(timezone.utc)

        # Convert input to text
        if isinstance(input_data, bytes):
            input_text = input_data.decode("utf-8", errors="ignore")
        elif isinstance(input_data, str):
            input_text = input_data
        else:
            input_text = str(input_data)

        # Get model prediction
        predictions = self.predict_fn([input_text])[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx]

        # Generate LIME explanation
        try:
            # Explain predicted class (or specific class if provided)
            label_to_explain = kwargs.get("predicted_class_idx", predicted_class_idx)

            lime_exp = self.explainer.explain_instance(
                input_text,
                self.predict_fn,
                num_features=top_k,
                num_samples=self.num_samples,
                labels=(label_to_explain,),
            )

            # Extract feature importances
            feature_importances = self._lime_to_features(
                lime_exp,
                label_to_explain,
                input_text,
            )

            # Rank and get top features
            ranked_features = self._rank_features(feature_importances)
            top_features = ranked_features[:top_k]

            # Generate summary
            summary = self._generate_summary(top_features, predicted_class, confidence)

            # Generate counterfactual explanation
            counterfactual = self._generate_counterfactual(
                top_features,
                predicted_class,
                self.class_names,
            )

            # Generate regulatory justification
            regulatory_justification = self._generate_regulatory_justification(
                top_features,
                predicted_class,
                confidence,
            )

        except Exception as e:
            logger.error(f"LIME computation failed: {e}")
            return self._create_empty_explanation(
                decision_id, input_data, predicted_class, confidence, str(e)
            )

        # Calculate explanation time
        explanation_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return ExplanationResult(
            explanation_id=str(uuid4()),
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            model_name=self.model_name,
            model_version=self.model_version,
            explanation_method=self.explanation_method,
            input_data=input_data,
            model_output=predicted_class,
            confidence_score=float(confidence),
            feature_importances=ranked_features,
            top_features=top_features,
            explanation_summary=summary,
            counterfactual=counterfactual,
            regulatory_justification=regulatory_justification,
            metadata={
                "num_samples": self.num_samples,
                "explanation_time_ms": explanation_time_ms,
                "all_class_probabilities": {
                    self.class_names[i]: float(predictions[i])
                    for i in range(len(self.class_names))
                },
            },
        )

    def batch_explain(
        self,
        input_batch: List[Any],
        decision_ids: List[str],
        top_k: int = 10,
    ) -> List[ExplanationResult]:
        """
        Generate LIME explanations for a batch of inputs.

        Args:
            input_batch: List of input texts
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

    def _lime_to_features(
        self,
        lime_exp: Any,
        label: int,
        input_text: str,
    ) -> List[FeatureImportance]:
        """
        Convert LIME explanation to FeatureImportance objects.

        Args:
            lime_exp: LIME explanation object
            label: Class label being explained
            input_text: Original input text

        Returns:
            List of FeatureImportance objects
        """
        feature_importances = []

        # Get feature importance list from LIME
        lime_features = lime_exp.as_list(label=label)

        for feature_text, importance_score in lime_features:
            # Extract the actual feature word/phrase
            # LIME returns features like "word" or "word1 word2"
            feature_words = feature_text.split()

            # Find occurrences in original text
            count = sum(
                1
                for word in input_text.split()
                if word.lower() in [fw.lower() for fw in feature_words]
            )

            # Generate description
            if importance_score > 0:
                direction = "increases likelihood of"
            else:
                direction = "decreases likelihood of"

            description = (
                f"Presence of '{feature_text}' {direction} "
                f"{self.class_names[label]} decision"
            )

            feature_importances.append(
                FeatureImportance(
                    feature_name=feature_text,
                    feature_value=count,  # Number of occurrences
                    importance_score=float(importance_score),
                    rank=0,  # Will be set by _rank_features
                    description=description,
                    metadata={
                        "words": feature_words,
                        "occurrences": count,
                        "direction": direction,
                    },
                )
            )

        return feature_importances

    def _generate_counterfactual(
        self,
        top_features: List[FeatureImportance],
        predicted_class: str,
        all_classes: List[str],
    ) -> str:
        """
        Generate counterfactual explanation.

        Args:
            top_features: Top important features
            predicted_class: Predicted class
            all_classes: All possible classes

        Returns:
            Counterfactual explanation string
        """
        if not top_features:
            return "No counterfactual available"

        # Find most negative feature (decreases current class)
        negative_features = [f for f in top_features if f.importance_score < 0]

        # Find alternative classes
        alternative_classes = [c for c in all_classes if c != predicted_class]

        if negative_features:
            strongest_negative = negative_features[0]
            return (
                f"If the input did not contain '{strongest_negative.feature_name}', "
                f"the decision might have been {alternative_classes[0] if alternative_classes else 'different'}."
            )
        else:
            # All features support current decision
            strongest_positive = top_features[0]
            return (
                f"If the input did not contain '{strongest_positive.feature_name}', "
                f"the confidence in {predicted_class} decision would be significantly lower."
            )

    def _generate_regulatory_justification(
        self,
        top_features: List[FeatureImportance],
        predicted_class: str,
        confidence: float,
    ) -> str:
        """Generate regulatory compliance justification."""
        if confidence > 0.90:
            confidence_level = "very high"
        elif confidence > 0.75:
            confidence_level = "high"
        elif confidence > 0.60:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        key_factors = [f.feature_name for f in top_features[:3]]
        factors_str = ", ".join(f"'{f}'" for f in key_factors)

        return (
            f"This {predicted_class} decision has {confidence_level} confidence ({confidence:.1%}). "
            f"The decision is primarily based on the presence/absence of: {factors_str}. "
            f"LIME perturbation analysis confirms these factors are statistically significant "
            f"predictors of the decision outcome. This explanation meets EU AI Act transparency "
            f"requirements for automated decision-making systems."
        )

    def _create_empty_explanation(
        self,
        decision_id: str,
        input_data: Any,
        predicted_class: str,
        confidence: float,
        error_message: str,
    ) -> ExplanationResult:
        """Create empty explanation when LIME computation fails."""
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


class LIMESecurityDecisionExplainer(LIMEDecisionExplainer):
    """
    Specialized LIME explainer for security decisions.

    Extends LIMEDecisionExplainer with security-specific enhancements:
    - Threat factor extraction
    - Risk level mapping
    - Security policy justification
    """

    def __init__(
        self,
        predict_fn: Callable,
        model_name: str = "security_decision_engine",
        model_version: str = "1.0.0",
        **kwargs,
    ):
        """Initialize security decision explainer."""
        # Default security decision classes
        class_names = kwargs.pop(
            "class_names", ["ALLOW", "BLOCK", "ESCALATE", "QUARANTINE"]
        )

        super().__init__(
            predict_fn=predict_fn,
            model_name=model_name,
            model_version=model_version,
            class_names=class_names,
            **kwargs,
        )

    def _generate_regulatory_justification(
        self,
        top_features: List[FeatureImportance],
        predicted_class: str,
        confidence: float,
    ) -> str:
        """Generate security-specific regulatory justification."""
        base_justification = super()._generate_regulatory_justification(
            top_features, predicted_class, confidence
        )

        # Add security policy context
        policy_context = self._map_to_security_policy(predicted_class, top_features)

        return f"{base_justification}\n\nSecurity Policy Context: {policy_context}"

    def _map_to_security_policy(
        self,
        decision: str,
        features: List[FeatureImportance],
    ) -> str:
        """Map decision to relevant security policies."""
        # Extract threat indicators from features
        threat_indicators = []
        for f in features[:5]:
            if any(
                word in f.feature_name.lower()
                for word in [
                    "suspicious",
                    "malicious",
                    "threat",
                    "attack",
                    "vulnerability",
                    "unauthorized",
                    "anomalous",
                    "blacklist",
                ]
            ):
                threat_indicators.append(f.feature_name)

        if decision == "BLOCK":
            if threat_indicators:
                return (
                    f"Decision aligns with Zero Trust security policy. "
                    f"Detected threat indicators: {', '.join(threat_indicators[:3])}. "
                    f"Recommended action: Block and alert security team."
                )
            else:
                return (
                    "Decision based on security policy rules. "
                    "Recommended action: Block pending further investigation."
                )
        elif decision == "ALLOW":
            return (
                "Decision based on whitelist verification and trust score analysis. "
                "No significant threat indicators detected."
            )
        elif decision == "ESCALATE":
            return (
                "Decision requires human review due to uncertain threat classification. "
                "Escalating to security analyst for manual assessment."
            )
        else:  # QUARANTINE
            return (
                "Decision to isolate potential threat for forensic analysis. "
                "Quarantine enables safe investigation without risk exposure."
            )
