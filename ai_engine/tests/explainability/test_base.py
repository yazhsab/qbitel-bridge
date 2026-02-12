"""
Tests for base explainability classes and interfaces.
"""

import pytest
from datetime import datetime
from ai_engine.explainability.base import (
    BaseExplainer,
    ExplanationResult,
    ExplanationType,
    FeatureImportance,
    FeatureImportanceMethod,
    ExplainerRegistry,
)


class MockExplainer(BaseExplainer):
    """Mock explainer for testing."""

    def _get_explanation_method(self) -> ExplanationType:
        return ExplanationType.RULE_BASED

    def explain(self, input_data, decision_id, top_k=10, **kwargs):
        return ExplanationResult(
            explanation_id="test-exp-001",
            decision_id=decision_id,
            timestamp=datetime.utcnow(),
            model_name=self.model_name,
            model_version=self.model_version,
            explanation_method=self.explanation_method,
            input_data=input_data,
            model_output="HTTP",
            confidence_score=0.95,
            feature_importances=[
                FeatureImportance("byte_0", 71, 0.15, 1, "G character"),
                FeatureImportance("byte_1", 69, 0.12, 2, "E character"),
            ],
            top_features=[FeatureImportance("byte_0", 71, 0.15, 1, "G character")],
            explanation_summary="Test explanation",
        )

    def batch_explain(self, input_batch, decision_ids, top_k=10):
        return [self.explain(inp, dec_id, top_k) for inp, dec_id in zip(input_batch, decision_ids)]


class TestFeatureImportance:
    """Tests for FeatureImportance dataclass."""

    def test_feature_importance_creation(self):
        """Test creating FeatureImportance."""
        fi = FeatureImportance(
            feature_name="byte_0",
            feature_value=71,
            importance_score=0.15,
            rank=1,
            description="HTTP GET verb",
        )

        assert fi.feature_name == "byte_0"
        assert fi.feature_value == 71
        assert fi.importance_score == 0.15
        assert fi.rank == 1
        assert fi.description == "HTTP GET verb"

    def test_feature_importance_with_metadata(self):
        """Test FeatureImportance with metadata."""
        fi = FeatureImportance(
            feature_name="byte_0",
            feature_value=71,
            importance_score=0.15,
            rank=1,
            metadata={"hex_value": "0x47", "char": "G"},
        )

        assert fi.metadata["hex_value"] == "0x47"
        assert fi.metadata["char"] == "G"


class TestExplanationResult:
    """Tests for ExplanationResult dataclass."""

    def test_explanation_result_creation(self):
        """Test creating ExplanationResult."""
        timestamp = datetime.utcnow()
        result = ExplanationResult(
            explanation_id="exp-001",
            decision_id="dec-001",
            timestamp=timestamp,
            model_name="protocol_classifier",
            model_version="2.1.0",
            explanation_method=ExplanationType.SHAP,
            input_data=b"GET / HTTP/1.1",
            model_output="HTTP",
            confidence_score=0.95,
            feature_importances=[],
            top_features=[],
            explanation_summary="Test summary",
        )

        assert result.explanation_id == "exp-001"
        assert result.decision_id == "dec-001"
        assert result.model_name == "protocol_classifier"
        assert result.confidence_score == 0.95

    def test_explanation_result_to_dict(self):
        """Test converting ExplanationResult to dictionary."""
        timestamp = datetime.utcnow()
        result = ExplanationResult(
            explanation_id="exp-001",
            decision_id="dec-001",
            timestamp=timestamp,
            model_name="protocol_classifier",
            model_version="2.1.0",
            explanation_method=ExplanationType.SHAP,
            input_data=b"GET",
            model_output="HTTP",
            confidence_score=0.95,
            feature_importances=[
                FeatureImportance("byte_0", 71, 0.15, 1),
            ],
            top_features=[FeatureImportance("byte_0", 71, 0.15, 1)],
            explanation_summary="Test",
        )

        result_dict = result.to_dict()

        assert result_dict["explanation_id"] == "exp-001"
        assert result_dict["model_name"] == "protocol_classifier"
        assert result_dict["confidence_score"] == 0.95
        assert len(result_dict["feature_importances"]) == 1
        assert result_dict["feature_importances"][0]["feature_name"] == "byte_0"


class TestBaseExplainer:
    """Tests for BaseExplainer abstract class."""

    def test_explainer_initialization(self):
        """Test initializing a mock explainer."""
        explainer = MockExplainer(
            model="mock_model",
            model_name="test_model",
            model_version="1.0.0",
        )

        assert explainer.model == "mock_model"
        assert explainer.model_name == "test_model"
        assert explainer.model_version == "1.0.0"
        assert explainer.explanation_method == ExplanationType.RULE_BASED

    def test_explain(self):
        """Test explain method."""
        explainer = MockExplainer(
            model="mock_model",
            model_name="test_model",
            model_version="1.0.0",
        )

        result = explainer.explain(
            input_data=b"GET",
            decision_id="dec-001",
        )

        assert result.decision_id == "dec-001"
        assert result.confidence_score == 0.95
        assert len(result.feature_importances) == 2

    def test_batch_explain(self):
        """Test batch explain method."""
        explainer = MockExplainer(
            model="mock_model",
            model_name="test_model",
            model_version="1.0.0",
        )

        results = explainer.batch_explain(
            input_batch=[b"GET", b"POST"],
            decision_ids=["dec-001", "dec-002"],
        )

        assert len(results) == 2
        assert results[0].decision_id == "dec-001"
        assert results[1].decision_id == "dec-002"

    def test_rank_features(self):
        """Test feature ranking."""
        explainer = MockExplainer(
            model="mock_model",
            model_name="test_model",
            model_version="1.0.0",
        )

        features = [
            FeatureImportance("byte_0", 71, 0.05, 0),
            FeatureImportance("byte_1", 69, 0.15, 0),
            FeatureImportance("byte_2", 84, 0.10, 0),
        ]

        ranked = explainer._rank_features(features)

        assert ranked[0].feature_name == "byte_1"  # Highest importance
        assert ranked[0].rank == 1
        assert ranked[1].feature_name == "byte_2"
        assert ranked[1].rank == 2
        assert ranked[2].feature_name == "byte_0"
        assert ranked[2].rank == 3

    def test_generate_summary(self):
        """Test summary generation."""
        explainer = MockExplainer(
            model="mock_model",
            model_name="test_model",
            model_version="1.0.0",
        )

        top_features = [
            FeatureImportance("byte_0", 71, 0.15, 1, "G character"),
            FeatureImportance("byte_1", 69, 0.12, 2, "E character"),
        ]

        summary = explainer._generate_summary(
            top_features=top_features,
            model_output="HTTP",
            confidence=0.95,
        )

        assert "HTTP" in summary
        assert "95" in summary  # 95% confidence
        assert "byte_0" in summary


class TestExplainerRegistry:
    """Tests for ExplainerRegistry."""

    def test_registry_register_and_get(self):
        """Test registering and retrieving explainers."""
        registry = ExplainerRegistry()

        explainer = MockExplainer(
            model="mock_model",
            model_name="test_model",
            model_version="1.0.0",
        )

        registry.register("test_model", explainer)

        retrieved = registry.get("test_model")
        assert retrieved is explainer

    def test_registry_get_nonexistent(self):
        """Test getting non-existent explainer."""
        registry = ExplainerRegistry()

        retrieved = registry.get("nonexistent_model")
        assert retrieved is None

    def test_registry_list_models(self):
        """Test listing registered models."""
        registry = ExplainerRegistry()

        explainer1 = MockExplainer("model1", "model1", "1.0.0")
        explainer2 = MockExplainer("model2", "model2", "1.0.0")

        registry.register("model1", explainer1)
        registry.register("model2", explainer2)

        models = registry.list_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models
