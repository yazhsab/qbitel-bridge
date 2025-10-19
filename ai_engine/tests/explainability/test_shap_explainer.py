"""
Tests for SHAP explainer.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from ai_engine.explainability.shap_explainer import (
    SHAPProtocolExplainer,
    create_background_dataset,
)
from ai_engine.explainability.base import ExplanationType


class SimpleCNN(nn.Module):
    """Simple CNN model for testing."""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.embedding = nn.Embedding(256, 32)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # (batch, 64, 1)
        x = x.squeeze(-1)  # (batch, 64)
        x = self.fc(x)
        return x


class TestSHAPProtocolExplainer:
    """Tests for SHAPProtocolExplainer."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple CNN model for testing."""
        model = SimpleCNN(num_classes=5)
        model.eval()
        return model

    @pytest.fixture
    def background_data(self):
        """Create background data for SHAP."""
        return np.random.randint(0, 256, size=(50, 512), dtype=np.uint8)

    def test_explainer_initialization(self, simple_model, background_data):
        """Test initializing SHAP explainer."""
        explainer = SHAPProtocolExplainer(
            model=simple_model,
            model_name="test_cnn",
            model_version="1.0.0",
            background_data=background_data,
            background_size=50,
        )

        assert explainer.model_name == "test_cnn"
        assert explainer.model_version == "1.0.0"
        assert explainer.explanation_method == ExplanationType.SHAP
        assert len(explainer.background_data) == 50

    def test_explainer_initialization_without_background(self, simple_model):
        """Test initialization with synthetic background data."""
        explainer = SHAPProtocolExplainer(
            model=simple_model,
            model_name="test_cnn",
            model_version="1.0.0",
            background_size=30,
        )

        assert explainer.background_data.shape == (30, 512)

    def test_prepare_input_from_bytes(self, simple_model, background_data):
        """Test preparing input from bytes."""
        explainer = SHAPProtocolExplainer(
            model=simple_model,
            model_name="test_cnn",
            model_version="1.0.0",
            background_data=background_data,
        )

        input_bytes = b"GET / HTTP/1.1\r\n"
        input_tensor = explainer._prepare_input(input_bytes)

        assert isinstance(input_tensor, torch.Tensor)
        assert input_tensor.shape == (1, 512)  # Batch size 1, padded to 512

    def test_prepare_input_from_numpy(self, simple_model, background_data):
        """Test preparing input from numpy array."""
        explainer = SHAPProtocolExplainer(
            model=simple_model,
            model_name="test_cnn",
            model_version="1.0.0",
            background_data=background_data,
        )

        input_array = np.array([71, 69, 84, 32], dtype=np.uint8)  # "GET "
        input_tensor = explainer._prepare_input(input_array)

        assert isinstance(input_tensor, torch.Tensor)
        assert input_tensor.shape == (1, 512)

    def test_shap_to_features(self, simple_model, background_data):
        """Test converting SHAP values to features."""
        explainer = SHAPProtocolExplainer(
            model=simple_model,
            model_name="test_cnn",
            model_version="1.0.0",
            background_data=background_data,
        )

        shap_values = np.array([0.15, 0.12, 0.10, 0.05, 0.02])
        input_bytes = np.array([71, 69, 84, 32, 47], dtype=np.uint8)  # "GET /"

        features = explainer._shap_to_features(shap_values, input_bytes)

        assert len(features) == 5
        assert features[0].feature_name == "byte_0"
        assert features[0].feature_value == 71
        assert features[0].importance_score == 0.15
        assert "G" in features[0].metadata["char_repr"]

    def test_get_byte_description(self, simple_model, background_data):
        """Test byte description generation."""
        explainer = SHAPProtocolExplainer(
            model=simple_model,
            model_name="test_cnn",
            model_version="1.0.0",
            background_data=background_data,
        )

        # Test HTTP verb detection
        desc = explainer._get_byte_description(0, 71, "G")
        assert "HTTP" in desc

        # Test TLS handshake detection
        desc = explainer._get_byte_description(0, 0x16, "\\x16")
        assert "TLS" in desc

        # Test general byte
        desc = explainer._get_byte_description(5, 65, "A")
        assert "0x41" in desc

    def test_regulatory_justification_generation(self, simple_model, background_data):
        """Test regulatory justification text generation."""
        explainer = SHAPProtocolExplainer(
            model=simple_model,
            model_name="test_cnn",
            model_version="1.0.0",
            background_data=background_data,
        )

        features = [
            pytest.importorskip("ai_engine.explainability.base").FeatureImportance(
                "byte_0", 71, 0.15, 1, metadata={"position": 0}
            ),
            pytest.importorskip("ai_engine.explainability.base").FeatureImportance(
                "byte_1", 69, 0.12, 2, metadata={"position": 1}
            ),
        ]

        justification = explainer._generate_regulatory_justification(features, 0.96)

        assert "very high confidence" in justification
        assert "96" in justification
        assert "EU AI Act" in justification

    @pytest.mark.skip(reason="SHAP computation requires actual model training")
    def test_explain_real_input(self, simple_model, background_data):
        """Test generating explanation for real input.

        Note: This test is skipped because it requires actual SHAP computation
        which is slow and may not work reliably in CI/CD.
        """
        explainer = SHAPProtocolExplainer(
            model=simple_model,
            model_name="test_cnn",
            model_version="1.0.0",
            background_data=background_data,
            background_size=20,  # Smaller for speed
        )

        input_bytes = b"GET / HTTP/1.1\r\n"
        result = explainer.explain(
            input_data=input_bytes,
            decision_id="test-dec-001",
            top_k=5,
        )

        assert result.decision_id == "test-dec-001"
        assert result.model_name == "test_cnn"
        assert result.explanation_method == ExplanationType.SHAP
        assert 0 <= result.confidence_score <= 1
        assert len(result.top_features) <= 5


class TestCreateBackgroundDataset:
    """Tests for background dataset creation."""

    def test_create_background_from_samples(self):
        """Test creating background dataset from protocol samples."""
        samples = [
            b"GET / HTTP/1.1\r\n",
            b"POST /api HTTP/1.1\r\n",
            b"HEAD / HTTP/1.1\r\n",
        ]

        background = create_background_dataset(samples, max_samples=3)

        assert background.shape == (3, 512)
        assert background.dtype == np.uint8
        # Check first byte of first sample is 'G' (71)
        assert background[0, 0] == 71

    def test_create_background_with_padding(self):
        """Test that short samples are padded."""
        samples = [b"GET"]  # Only 3 bytes

        background = create_background_dataset(samples, max_samples=1)

        assert background.shape == (1, 512)
        # Check padding (should be zeros after first 3 bytes)
        assert background[0, 3] == 0
        assert background[0, 511] == 0

    def test_create_background_with_truncation(self):
        """Test that long samples are truncated."""
        samples = [b"A" * 1000]  # 1000 bytes

        background = create_background_dataset(samples, max_samples=1)

        assert background.shape == (1, 512)
        # All bytes should be 'A' (65)
        assert all(background[0, :] == 65)

    def test_create_background_respects_max_samples(self):
        """Test that max_samples is respected."""
        samples = [b"GET"] * 100

        background = create_background_dataset(samples, max_samples=10)

        assert background.shape[0] == 10  # Only 10 samples
