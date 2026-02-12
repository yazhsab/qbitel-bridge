"""
Tests for LIME Explainer.
"""

import pytest
import numpy as np

from ai_engine.explainability.lime_explainer import (
    LIMEDecisionExplainer,
    LIMESecurityDecisionExplainer,
)
from ai_engine.explainability.base import ExplanationType


class TestLIMEDecisionExplainer:
    """Tests for LIMEDecisionExplainer."""

    @pytest.fixture
    def mock_predict_fn(self):
        """Create mock prediction function."""

        def predict_fn(texts):
            """
            Mock LLM prediction function.
            Returns probabilities for ['ALLOW', 'BLOCK', 'ESCALATE']
            """
            results = []
            for text in texts:
                # Simple rule-based mock
                if "suspicious" in text.lower() or "malicious" in text.lower():
                    # High probability of BLOCK
                    results.append(np.array([0.1, 0.8, 0.1]))
                elif "authorized" in text.lower() or "verified" in text.lower():
                    # High probability of ALLOW
                    results.append(np.array([0.8, 0.1, 0.1]))
                else:
                    # Uncertain - ESCALATE
                    results.append(np.array([0.3, 0.3, 0.4]))
            return np.array(results)

        return predict_fn

    def test_initialization(self, mock_predict_fn):
        """Test LIME explainer initialization."""
        explainer = LIMEDecisionExplainer(
            predict_fn=mock_predict_fn,
            model_name="security_decision_engine",
            model_version="1.0.0",
            class_names=["ALLOW", "BLOCK", "ESCALATE"],
            num_samples=100,  # Reduce for faster tests
        )

        assert explainer.model_name == "security_decision_engine"
        assert explainer.model_version == "1.0.0"
        assert explainer.explanation_method == ExplanationType.LIME
        assert explainer.class_names == ["ALLOW", "BLOCK", "ESCALATE"]
        assert explainer.num_samples == 100

    def test_explain_suspicious_event(self, mock_predict_fn):
        """Test explaining a suspicious security event."""
        explainer = LIMEDecisionExplainer(
            predict_fn=mock_predict_fn,
            model_name="security_decision_engine",
            model_version="1.0.0",
            class_names=["ALLOW", "BLOCK", "ESCALATE"],
            num_samples=100,
        )

        event_text = "Suspicious network traffic detected from unknown source"
        explanation = explainer.explain(
            input_data=event_text,
            decision_id="sec-001",
            top_k=5,
        )

        assert explanation.decision_id == "sec-001"
        assert explanation.model_output == "BLOCK"
        assert explanation.confidence_score > 0.5
        assert len(explanation.top_features) <= 5
        assert explanation.explanation_summary is not None
        assert explanation.counterfactual is not None
        assert explanation.regulatory_justification is not None

        # Check that "suspicious" is in top features
        feature_names = [f.feature_name for f in explanation.top_features]
        assert any("suspicious" in name.lower() for name in feature_names)

    def test_explain_authorized_event(self, mock_predict_fn):
        """Test explaining an authorized event."""
        explainer = LIMEDecisionExplainer(
            predict_fn=mock_predict_fn,
            model_name="security_decision_engine",
            model_version="1.0.0",
            class_names=["ALLOW", "BLOCK", "ESCALATE"],
            num_samples=100,
        )

        event_text = "Authorized user from verified IP address"
        explanation = explainer.explain(
            input_data=event_text,
            decision_id="sec-002",
            top_k=5,
        )

        assert explanation.decision_id == "sec-002"
        assert explanation.model_output == "ALLOW"
        assert explanation.confidence_score > 0.5

    def test_batch_explain(self, mock_predict_fn):
        """Test batch explanation generation."""
        explainer = LIMEDecisionExplainer(
            predict_fn=mock_predict_fn,
            model_name="security_decision_engine",
            model_version="1.0.0",
            class_names=["ALLOW", "BLOCK", "ESCALATE"],
            num_samples=50,  # Smaller for speed
        )

        texts = [
            "Suspicious activity detected",
            "Authorized access granted",
        ]
        decision_ids = ["sec-001", "sec-002"]

        explanations = explainer.batch_explain(
            input_batch=texts,
            decision_ids=decision_ids,
            top_k=3,
        )

        assert len(explanations) == 2
        assert explanations[0].decision_id == "sec-001"
        assert explanations[1].decision_id == "sec-002"
        assert explanations[0].model_output == "BLOCK"
        assert explanations[1].model_output == "ALLOW"

    def test_explain_with_bytes_input(self, mock_predict_fn):
        """Test explaining with bytes input."""
        explainer = LIMEDecisionExplainer(
            predict_fn=mock_predict_fn,
            model_name="security_decision_engine",
            model_version="1.0.0",
            class_names=["ALLOW", "BLOCK", "ESCALATE"],
            num_samples=50,
        )

        event_bytes = b"Suspicious network activity"
        explanation = explainer.explain(
            input_data=event_bytes,
            decision_id="sec-003",
        )

        assert explanation.decision_id == "sec-003"
        assert explanation.model_output == "BLOCK"

    def test_counterfactual_generation(self, mock_predict_fn):
        """Test counterfactual explanation generation."""
        explainer = LIMEDecisionExplainer(
            predict_fn=mock_predict_fn,
            model_name="security_decision_engine",
            model_version="1.0.0",
            class_names=["ALLOW", "BLOCK", "ESCALATE"],
            num_samples=50,
        )

        event_text = "Suspicious malicious activity detected"
        explanation = explainer.explain(
            input_data=event_text,
            decision_id="sec-004",
        )

        # Should have counterfactual explaining alternative decision
        assert explanation.counterfactual is not None
        assert len(explanation.counterfactual) > 0
        assert "if" in explanation.counterfactual.lower()

    def test_regulatory_justification(self, mock_predict_fn):
        """Test regulatory justification generation."""
        explainer = LIMEDecisionExplainer(
            predict_fn=mock_predict_fn,
            model_name="security_decision_engine",
            model_version="1.0.0",
            class_names=["ALLOW", "BLOCK", "ESCALATE"],
            num_samples=50,
        )

        event_text = "Suspicious activity"
        explanation = explainer.explain(
            input_data=event_text,
            decision_id="sec-005",
        )

        # Check regulatory justification exists and mentions EU AI Act
        assert explanation.regulatory_justification is not None
        assert "EU AI Act" in explanation.regulatory_justification
        assert "confidence" in explanation.regulatory_justification.lower()

    def test_metadata_includes_explanation_time(self, mock_predict_fn):
        """Test that metadata includes explanation time."""
        explainer = LIMEDecisionExplainer(
            predict_fn=mock_predict_fn,
            model_name="security_decision_engine",
            model_version="1.0.0",
            class_names=["ALLOW", "BLOCK", "ESCALATE"],
            num_samples=50,
        )

        explanation = explainer.explain(
            input_data="Test event",
            decision_id="sec-006",
        )

        assert "explanation_time_ms" in explanation.metadata
        assert explanation.metadata["explanation_time_ms"] > 0
        assert "all_class_probabilities" in explanation.metadata


class TestLIMESecurityDecisionExplainer:
    """Tests for LIMESecurityDecisionExplainer."""

    @pytest.fixture
    def mock_security_predict_fn(self):
        """Create mock security prediction function."""

        def predict_fn(texts):
            results = []
            for text in texts:
                if "threat" in text.lower() or "attack" in text.lower():
                    results.append(np.array([0.05, 0.85, 0.05, 0.05]))  # BLOCK
                elif "safe" in text.lower():
                    results.append(np.array([0.80, 0.05, 0.10, 0.05]))  # ALLOW
                elif "quarantine" in text.lower():
                    results.append(np.array([0.05, 0.05, 0.05, 0.85]))  # QUARANTINE
                else:
                    results.append(np.array([0.15, 0.20, 0.50, 0.15]))  # ESCALATE
            return np.array(results)

        return predict_fn

    def test_security_explainer_initialization(self, mock_security_predict_fn):
        """Test security explainer initialization with default classes."""
        explainer = LIMESecurityDecisionExplainer(
            predict_fn=mock_security_predict_fn,
        )

        assert explainer.model_name == "security_decision_engine"
        assert explainer.class_names == ["ALLOW", "BLOCK", "ESCALATE", "QUARANTINE"]

    def test_security_policy_mapping_block(self, mock_security_predict_fn):
        """Test security policy mapping for BLOCK decision."""
        explainer = LIMESecurityDecisionExplainer(
            predict_fn=mock_security_predict_fn,
            num_samples=50,
        )

        event_text = "Threat detected with attack signatures"
        explanation = explainer.explain(
            input_data=event_text,
            decision_id="sec-007",
        )

        assert explanation.model_output == "BLOCK"
        # Check that regulatory justification includes security policy context
        assert "Security Policy Context" in explanation.regulatory_justification
        assert "Zero Trust" in explanation.regulatory_justification or "threat" in explanation.regulatory_justification.lower()

    def test_security_policy_mapping_allow(self, mock_security_predict_fn):
        """Test security policy mapping for ALLOW decision."""
        explainer = LIMESecurityDecisionExplainer(
            predict_fn=mock_security_predict_fn,
            num_samples=50,
        )

        event_text = "Safe verified connection from whitelist"
        explanation = explainer.explain(
            input_data=event_text,
            decision_id="sec-008",
        )

        assert explanation.model_output == "ALLOW"
        assert "Security Policy Context" in explanation.regulatory_justification
        assert (
            "whitelist" in explanation.regulatory_justification.lower() or "no" in explanation.regulatory_justification.lower()
        )

    def test_security_policy_mapping_escalate(self, mock_security_predict_fn):
        """Test security policy mapping for ESCALATE decision."""
        explainer = LIMESecurityDecisionExplainer(
            predict_fn=mock_security_predict_fn,
            num_samples=50,
        )

        event_text = "Uncertain network behavior requires review"
        explanation = explainer.explain(
            input_data=event_text,
            decision_id="sec-009",
        )

        assert explanation.model_output == "ESCALATE"
        assert "Security Policy Context" in explanation.regulatory_justification
        assert (
            "human review" in explanation.regulatory_justification.lower()
            or "manual" in explanation.regulatory_justification.lower()
        )

    def test_security_policy_mapping_quarantine(self, mock_security_predict_fn):
        """Test security policy mapping for QUARANTINE decision."""
        explainer = LIMESecurityDecisionExplainer(
            predict_fn=mock_security_predict_fn,
            num_samples=50,
        )

        event_text = "Potential malware detected, quarantine recommended"
        explanation = explainer.explain(
            input_data=event_text,
            decision_id="sec-010",
        )

        assert explanation.model_output == "QUARANTINE"
        assert "Security Policy Context" in explanation.regulatory_justification
        assert (
            "quarantine" in explanation.regulatory_justification.lower()
            or "isolate" in explanation.regulatory_justification.lower()
        )
