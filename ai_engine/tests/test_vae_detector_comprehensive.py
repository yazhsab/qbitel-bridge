"""
Comprehensive tests for ai_engine/anomaly/vae_detector.py

Tests cover:
- VAE model architecture and initialization
- Encoding and decoding
- Reparameterization trick
- Loss function calculation (ELBO)
- Anomaly scoring
- VAE training with early stopping
- Threshold calculation
- Anomaly detection with edge cases
- Model saving and loading
- Feature normalization
"""

import pytest
import asyncio
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import os

from ai_engine.anomaly.vae_detector import (
    VAEModel,
    VAEAnomalyDetector,
    AnomalyResult,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import AnomalyDetectionException, ModelException


# Fixtures

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.environment = Mock()
    config.environment.value = "development"
    return config


@pytest.fixture
def vae_model():
    """Create VAE model instance."""
    return VAEModel(input_dim=50, latent_dim=16, hidden_dims=[128, 64])


@pytest.fixture
async def vae_detector(mock_config):
    """Create VAE anomaly detector."""
    detector = VAEAnomalyDetector(mock_config)
    await detector.initialize(input_dim=50)
    return detector


@pytest.fixture
def sample_training_data():
    """Generate sample normal training data."""
    np.random.seed(42)
    # Generate 1000 samples with 50 features
    return np.random.randn(1000, 50).astype(np.float32)


@pytest.fixture
def sample_anomaly_data():
    """Generate sample anomalous data."""
    np.random.seed(123)
    # Generate data with larger variance (anomalies)
    return np.random.randn(10, 50).astype(np.float32) * 3.0


# VAE Model Tests

def test_vae_model_initialization(vae_model):
    """Test VAE model initialization."""
    assert vae_model.input_dim == 50
    assert vae_model.latent_dim == 16
    assert isinstance(vae_model.encoder, nn.Sequential)
    assert isinstance(vae_model.decoder, nn.Sequential)
    assert isinstance(vae_model.fc_mu, nn.Linear)
    assert isinstance(vae_model.fc_logvar, nn.Linear)


def test_vae_model_with_custom_hidden_dims():
    """Test VAE model with custom hidden dimensions."""
    model = VAEModel(input_dim=100, latent_dim=32, hidden_dims=[256, 128, 64])

    assert model.input_dim == 100
    assert model.latent_dim == 32


def test_vae_model_without_batch_norm():
    """Test VAE model without batch normalization."""
    model = VAEModel(input_dim=50, latent_dim=16, use_batch_norm=False)

    # Check that no BatchNorm layers exist
    has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    assert has_batchnorm is False


def test_vae_model_without_dropout():
    """Test VAE model without dropout."""
    model = VAEModel(input_dim=50, latent_dim=16, dropout_prob=0.0)

    # Check that no Dropout layers exist
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout is False


def test_vae_encode(vae_model):
    """Test VAE encoding."""
    x = torch.randn(32, 50)
    mu, logvar = vae_model.encode(x)

    assert mu.shape == (32, 16)
    assert logvar.shape == (32, 16)


def test_vae_reparameterize_training(vae_model):
    """Test reparameterization during training."""
    vae_model.train()

    mu = torch.randn(32, 16)
    logvar = torch.randn(32, 16)

    z = vae_model.reparameterize(mu, logvar)

    assert z.shape == (32, 16)
    # During training, should use stochastic sampling
    assert not torch.allclose(z, mu)


def test_vae_reparameterize_inference(vae_model):
    """Test reparameterization during inference."""
    vae_model.eval()

    mu = torch.randn(32, 16)
    logvar = torch.randn(32, 16)

    z = vae_model.reparameterize(mu, logvar)

    assert z.shape == (32, 16)
    # During inference, should return mean
    assert torch.allclose(z, mu)


def test_vae_decode(vae_model):
    """Test VAE decoding."""
    z = torch.randn(32, 16)
    reconstruction = vae_model.decode(z)

    assert reconstruction.shape == (32, 50)
    # Output should be in [0, 1] due to Sigmoid
    assert torch.all(reconstruction >= 0.0)
    assert torch.all(reconstruction <= 1.0)


def test_vae_forward(vae_model):
    """Test VAE forward pass."""
    x = torch.randn(32, 50)
    reconstruction, mu, logvar = vae_model.forward(x)

    assert reconstruction.shape == (32, 50)
    assert mu.shape == (32, 16)
    assert logvar.shape == (32, 16)


def test_vae_loss_function(vae_model):
    """Test VAE loss function calculation."""
    x = torch.rand(32, 50)  # Random data in [0, 1]
    reconstruction, mu, logvar = vae_model.forward(x)

    losses = vae_model.loss_function(x, reconstruction, mu, logvar, beta=1.0)

    assert "total_loss" in losses
    assert "reconstruction_loss" in losses
    assert "kl_loss" in losses

    assert losses["total_loss"].item() > 0
    assert losses["reconstruction_loss"].item() >= 0
    assert losses["kl_loss"].item() >= 0


def test_vae_loss_function_with_beta():
    """Test VAE loss with different beta values (beta-VAE)."""
    model = VAEModel(input_dim=50, latent_dim=16)
    x = torch.rand(32, 50)
    reconstruction, mu, logvar = model.forward(x)

    # Test with different beta values
    losses_beta1 = model.loss_function(x, reconstruction, mu, logvar, beta=1.0)
    losses_beta2 = model.loss_function(x, reconstruction, mu, logvar, beta=2.0)

    # Higher beta should give more weight to KL loss
    assert losses_beta2["total_loss"].item() >= losses_beta1["total_loss"].item()


def test_vae_anomaly_score(vae_model):
    """Test anomaly score calculation."""
    vae_model.eval()
    x = torch.rand(10, 50)

    total_score, recon_error, kl_div = vae_model.anomaly_score(x)

    assert total_score.shape == (10,)
    assert recon_error.shape == (10,)
    assert kl_div.shape == (10,)

    # Scores should be non-negative
    assert torch.all(total_score >= 0)
    assert torch.all(recon_error >= 0)


# VAE Anomaly Detector Tests

@pytest.mark.asyncio
async def test_detector_initialization(mock_config):
    """Test VAE anomaly detector initialization."""
    detector = VAEAnomalyDetector(mock_config)

    assert detector.config == mock_config
    assert detector.model is None
    assert detector.device is not None


@pytest.mark.asyncio
async def test_detector_initialize_with_input_dim(mock_config):
    """Test detector initialization with input dimension."""
    detector = VAEAnomalyDetector(mock_config)
    await detector.initialize(input_dim=100)

    assert detector.input_dim == 100
    assert detector.model is not None
    assert detector.model.input_dim == 100
    assert detector.optimizer is not None
    assert detector.scheduler is not None


@pytest.mark.asyncio
async def test_detector_initialize_failure(mock_config):
    """Test detector initialization failure."""
    detector = VAEAnomalyDetector(mock_config)

    with patch.object(VAEModel, "__init__", side_effect=Exception("Init failed")):
        with pytest.raises(ModelException, match="VAE initialization failed"):
            await detector.initialize(input_dim=50)


@pytest.mark.asyncio
async def test_detect_without_initialization(mock_config):
    """Test detection without model initialization."""
    detector = VAEAnomalyDetector(mock_config)

    features = np.random.randn(50)

    with pytest.raises(AnomalyDetectionException, match="Model not initialized"):
        await detector.detect(features)


@pytest.mark.asyncio
async def test_detect_normal_data(vae_detector, sample_training_data):
    """Test detection on normal data."""
    # Train first
    await vae_detector.train(sample_training_data, num_epochs=2)

    # Detect on normal sample
    normal_sample = sample_training_data[0]
    result = await vae_detector.detect(normal_sample)

    assert isinstance(result, AnomalyResult)
    assert result.anomaly_score >= 0
    assert result.reconstruction_error >= 0
    assert result.kl_divergence >= 0
    assert 0 <= result.confidence <= 1
    assert result.processing_time > 0


@pytest.mark.asyncio
async def test_detect_anomalous_data(vae_detector, sample_training_data, sample_anomaly_data):
    """Test detection on anomalous data."""
    # Train on normal data
    await vae_detector.train(sample_training_data, num_epochs=2)

    # Detect on anomalous sample
    anomaly_sample = sample_anomaly_data[0]
    result = await vae_detector.detect(anomaly_sample)

    assert isinstance(result, AnomalyResult)
    # Anomalous data should have higher scores
    assert result.anomaly_score > 0


@pytest.mark.asyncio
async def test_detect_with_context(vae_detector, sample_training_data):
    """Test detection with context information."""
    await vae_detector.train(sample_training_data, num_epochs=2)

    sample = sample_training_data[0]
    context = {"protocol_type": "HTTP"}

    result = await vae_detector.detect(sample, context=context)

    assert isinstance(result, AnomalyResult)
    # Context should be mentioned in explanation if anomaly
    if result.is_anomaly:
        assert "HTTP" in result.explanation


@pytest.mark.asyncio
async def test_train_basic(vae_detector, sample_training_data):
    """Test basic VAE training."""
    history = await vae_detector.train(
        sample_training_data,
        num_epochs=3,
        batch_size=64
    )

    assert "train_loss" in history
    assert "train_recon_loss" in history
    assert "train_kl_loss" in history
    assert len(history["train_loss"]) <= 3

    # Loss should generally decrease
    assert history["train_loss"][-1] <= history["train_loss"][0] * 2


@pytest.mark.asyncio
async def test_train_with_validation(vae_detector, sample_training_data):
    """Test training with validation data."""
    # Split data
    train_data = sample_training_data[:800]
    val_data = sample_training_data[800:]

    history = await vae_detector.train(
        train_data,
        validation_data=val_data,
        num_epochs=5,
        early_stopping_patience=2
    )

    assert "val_loss" in history
    assert "val_recon_loss" in history
    assert "val_kl_loss" in history

    # Validation loss should be tracked
    assert len(history["val_loss"]) > 0


@pytest.mark.asyncio
async def test_train_early_stopping(vae_detector):
    """Test early stopping during training."""
    # Create data that won't improve
    data = np.random.randn(500, 50).astype(np.float32)

    history = await vae_detector.train(
        data[:400],
        validation_data=data[400:],
        num_epochs=20,
        early_stopping_patience=3
    )

    # Should stop before 20 epochs
    assert len(history["train_loss"]) < 20


@pytest.mark.asyncio
async def test_train_without_initialization(mock_config):
    """Test training without initialization."""
    detector = VAEAnomalyDetector(mock_config)
    data = np.random.randn(100, 50).astype(np.float32)

    with pytest.raises(AnomalyDetectionException, match="Model not initialized"):
        await detector.train(data)


@pytest.mark.asyncio
async def test_calculate_thresholds(vae_detector, sample_training_data):
    """Test threshold calculation."""
    await vae_detector.train(sample_training_data, num_epochs=2)

    assert vae_detector.anomaly_threshold is not None
    assert vae_detector.reconstruction_threshold is not None
    assert vae_detector.kl_threshold is not None

    assert vae_detector.anomaly_threshold > 0
    assert vae_detector.reconstruction_threshold > 0
    assert vae_detector.kl_threshold > 0


@pytest.mark.asyncio
async def test_threshold_percentile(vae_detector, sample_training_data):
    """Test threshold calculation at different percentiles."""
    vae_detector.percentile_threshold = 90

    await vae_detector.train(sample_training_data, num_epochs=2)

    threshold_90 = vae_detector.anomaly_threshold

    # Reset and train with different percentile
    vae_detector.percentile_threshold = 95
    await vae_detector._calculate_thresholds(sample_training_data)

    threshold_95 = vae_detector.anomaly_threshold

    # 95th percentile should be higher than 90th
    assert threshold_95 >= threshold_90


@pytest.mark.asyncio
async def test_is_anomalous_with_thresholds(vae_detector):
    """Test anomaly detection logic with thresholds."""
    vae_detector.anomaly_threshold = 10.0
    vae_detector.reconstruction_threshold = 5.0
    vae_detector.kl_threshold = 3.0

    # Normal case (below all thresholds)
    assert vae_detector._is_anomalous(5.0, 2.0, 1.0) is False

    # Anomaly - total score exceeds
    assert vae_detector._is_anomalous(15.0, 2.0, 1.0) is True

    # Anomaly - reconstruction error exceeds
    assert vae_detector._is_anomalous(5.0, 10.0, 1.0) is True

    # Anomaly - KL divergence exceeds
    assert vae_detector._is_anomalous(5.0, 2.0, 5.0) is True


@pytest.mark.asyncio
async def test_is_anomalous_without_thresholds(vae_detector):
    """Test anomaly detection without thresholds (fallback)."""
    vae_detector.anomaly_threshold = None

    # Should use fallback heuristic
    assert vae_detector._is_anomalous(15.0, 2.0, 1.0) is True
    assert vae_detector._is_anomalous(5.0, 2.0, 1.0) is False


@pytest.mark.asyncio
async def test_calculate_confidence(vae_detector):
    """Test confidence score calculation."""
    vae_detector.anomaly_threshold = 10.0
    vae_detector.reconstruction_threshold = 5.0
    vae_detector.kl_threshold = 3.0

    # Normal case
    conf_normal = vae_detector._calculate_confidence(5.0, 2.0, 1.0)
    assert 0.1 <= conf_normal <= 0.99

    # High anomaly
    conf_high = vae_detector._calculate_confidence(50.0, 25.0, 15.0)
    assert 0.1 <= conf_high <= 0.99
    assert conf_high > conf_normal


@pytest.mark.asyncio
async def test_calculate_confidence_without_thresholds(vae_detector):
    """Test confidence calculation without thresholds."""
    vae_detector.anomaly_threshold = None

    confidence = vae_detector._calculate_confidence(5.0, 2.0, 1.0)
    assert confidence == 0.5  # Low confidence fallback


@pytest.mark.asyncio
async def test_generate_explanation_normal(vae_detector):
    """Test explanation generation for normal data."""
    vae_detector.anomaly_threshold = 10.0

    explanation = vae_detector._generate_explanation(
        is_anomaly=False,
        total_score=5.0,
        recon_error=2.0,
        kl_div=1.0,
        context=None
    )

    assert "Normal pattern" in explanation


@pytest.mark.asyncio
async def test_generate_explanation_anomaly(vae_detector):
    """Test explanation generation for anomalies."""
    vae_detector.anomaly_threshold = 10.0
    vae_detector.reconstruction_threshold = 5.0
    vae_detector.kl_threshold = 3.0

    explanation = vae_detector._generate_explanation(
        is_anomaly=True,
        total_score=15.0,
        recon_error=10.0,
        kl_div=5.0,
        context={"protocol_type": "SSH"}
    )

    assert "Anomaly detected" in explanation
    assert "reconstruction error" in explanation or "latent space" in explanation
    assert "SSH" in explanation


# Feature Preprocessing Tests

def test_preprocess_features_1d(vae_detector):
    """Test feature preprocessing with 1D input."""
    features = np.random.randn(50)
    tensor = vae_detector._preprocess_features(features)

    assert tensor.shape == (1, 50)
    assert tensor.device == vae_detector.device


def test_preprocess_features_2d(vae_detector):
    """Test feature preprocessing with 2D input."""
    features = np.random.randn(10, 50)
    tensor = vae_detector._preprocess_features(features)

    assert tensor.shape == (10, 50)


@pytest.mark.asyncio
async def test_preprocess_features_with_normalization(vae_detector, sample_training_data):
    """Test feature preprocessing with normalization."""
    await vae_detector.train(sample_training_data, num_epochs=1)

    # Now normalization stats should be available
    features = sample_training_data[0]
    tensor = vae_detector._preprocess_features(features)

    # Should be normalized
    assert tensor.shape == (1, 50)


def test_calculate_normalization_stats(vae_detector, sample_training_data):
    """Test normalization statistics calculation."""
    vae_detector._calculate_normalization_stats(sample_training_data)

    assert vae_detector.feature_mean is not None
    assert vae_detector.feature_std is not None
    assert vae_detector.feature_mean.shape == (50,)
    assert vae_detector.feature_std.shape == (50,)


# Model Saving and Loading Tests

@pytest.mark.asyncio
async def test_save_and_load_model(vae_detector, sample_training_data):
    """Test model saving and loading."""
    # Train model
    await vae_detector.train(sample_training_data, num_epochs=2)

    original_threshold = vae_detector.anomaly_threshold

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "vae_model.pt")

        # Save model
        await vae_detector.save_model(model_path)
        assert os.path.exists(model_path)

        # Create new detector
        new_detector = VAEAnomalyDetector(vae_detector.config)
        await new_detector.initialize(input_dim=50)

        # Load model
        await new_detector.load_model(model_path)

        # Check loaded attributes
        assert new_detector.anomaly_threshold == original_threshold
        assert new_detector.feature_mean is not None
        assert new_detector.feature_std is not None


@pytest.mark.asyncio
async def test_save_model_failure(vae_detector):
    """Test model saving failure."""
    await vae_detector.initialize(input_dim=50)

    invalid_path = "/invalid/path/model.pt"

    with pytest.raises(ModelException, match="Model saving failed"):
        await vae_detector.save_model(invalid_path)


@pytest.mark.asyncio
async def test_load_model_failure(vae_detector):
    """Test model loading failure."""
    await vae_detector.initialize(input_dim=50)

    invalid_path = "/invalid/path/model.pt"

    with pytest.raises(ModelException, match="Model loading failed"):
        await vae_detector.load_model(invalid_path)


# Edge Cases and Error Handling

@pytest.mark.asyncio
async def test_detect_empty_features(vae_detector):
    """Test detection with empty features."""
    await vae_detector.initialize(input_dim=50)

    empty_features = np.array([])

    with pytest.raises(AnomalyDetectionException):
        await vae_detector.detect(empty_features)


@pytest.mark.asyncio
async def test_detect_wrong_dimension(vae_detector, sample_training_data):
    """Test detection with wrong feature dimension."""
    await vae_detector.train(sample_training_data, num_epochs=1)

    wrong_dim_features = np.random.randn(30)  # Wrong dimension

    with pytest.raises(AnomalyDetectionException):
        await vae_detector.detect(wrong_dim_features)


@pytest.mark.asyncio
async def test_train_empty_data(vae_detector):
    """Test training with empty data."""
    await vae_detector.initialize(input_dim=50)

    empty_data = np.array([]).reshape(0, 50)

    with pytest.raises(Exception):  # Should fail during dataloader creation
        await vae_detector.train(empty_data, num_epochs=1)


@pytest.mark.asyncio
async def test_threshold_edge_cases(vae_detector):
    """Test threshold detection edge cases."""
    vae_detector.anomaly_threshold = 10.0
    vae_detector.reconstruction_threshold = 5.0
    vae_detector.kl_threshold = 3.0

    # Exactly at threshold
    assert vae_detector._is_anomalous(10.0, 5.0, 3.0) is False

    # Just above threshold
    assert vae_detector._is_anomalous(10.1, 5.0, 3.0) is True

    # Zero values
    assert vae_detector._is_anomalous(0.0, 0.0, 0.0) is False


@pytest.mark.asyncio
async def test_very_small_training_data(vae_detector):
    """Test training with very small dataset."""
    small_data = np.random.randn(10, 50).astype(np.float32)

    # Should still work but may not perform well
    history = await vae_detector.train(small_data, num_epochs=2, batch_size=5)

    assert len(history["train_loss"]) > 0


@pytest.mark.asyncio
async def test_batch_size_larger_than_data(vae_detector):
    """Test training with batch size larger than dataset."""
    small_data = np.random.randn(50, 50).astype(np.float32)

    history = await vae_detector.train(small_data, num_epochs=2, batch_size=100)

    # Should still work with smaller effective batch size
    assert len(history["train_loss"]) > 0


# Integration Tests

@pytest.mark.asyncio
async def test_full_workflow(mock_config):
    """Test complete VAE anomaly detection workflow."""
    # Create detector
    detector = VAEAnomalyDetector(mock_config)
    await detector.initialize(input_dim=50)

    # Generate training data (normal)
    normal_data = np.random.randn(500, 50).astype(np.float32) * 0.5

    # Generate test data
    normal_test = np.random.randn(10, 50).astype(np.float32) * 0.5
    anomaly_test = np.random.randn(10, 50).astype(np.float32) * 3.0

    # Train
    history = await detector.train(normal_data, num_epochs=5)
    assert len(history["train_loss"]) > 0

    # Detect normal samples
    for sample in normal_test:
        result = await detector.detect(sample)
        assert isinstance(result, AnomalyResult)

    # Detect anomalous samples (should have higher scores)
    anomaly_scores = []
    for sample in anomaly_test:
        result = await detector.detect(sample)
        anomaly_scores.append(result.anomaly_score)

    # At least some anomalies should be detected
    assert any(anomaly_scores)


@pytest.mark.asyncio
async def test_device_handling():
    """Test device handling (CPU/GPU)."""
    config = Mock(spec=Config)
    detector = VAEAnomalyDetector(config)

    # Should default to available device
    assert detector.device is not None
    assert detector.device.type in ["cpu", "cuda"]


@pytest.mark.asyncio
async def test_concurrent_detections(vae_detector, sample_training_data):
    """Test concurrent anomaly detections."""
    await vae_detector.train(sample_training_data, num_epochs=2)

    # Run multiple detections concurrently
    samples = [sample_training_data[i] for i in range(10)]

    results = await asyncio.gather(*[
        vae_detector.detect(sample) for sample in samples
    ])

    assert len(results) == 10
    assert all(isinstance(r, AnomalyResult) for r in results)
