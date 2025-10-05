"""
Tests for ai_engine/anomaly/vae_detector.py
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path


class TestVAEModel:
    """Test suite for VAEModel."""

    @pytest.fixture
    def vae_model(self):
        """Create VAE model instance."""
        from ai_engine.anomaly.vae_detector import VAEModel

        return VAEModel(input_dim=50, latent_dim=16, hidden_dims=[128, 64])

    def test_vae_model_initialization(self, vae_model):
        """Test VAE model initialization."""
        assert vae_model.input_dim == 50
        assert vae_model.latent_dim == 16
        assert vae_model.use_batch_norm is True

    def test_vae_model_encode(self, vae_model):
        """Test VAE encoding."""
        x = torch.randn(10, 50)
        mu, logvar = vae_model.encode(x)

        assert mu.shape == (10, 16)
        assert logvar.shape == (10, 16)

    def test_vae_model_reparameterize_training(self, vae_model):
        """Test reparameterization during training."""
        vae_model.train()
        mu = torch.randn(10, 16)
        logvar = torch.randn(10, 16)

        z = vae_model.reparameterize(mu, logvar)

        assert z.shape == (10, 16)

    def test_vae_model_reparameterize_inference(self, vae_model):
        """Test reparameterization during inference."""
        vae_model.eval()
        mu = torch.randn(10, 16)
        logvar = torch.randn(10, 16)

        z = vae_model.reparameterize(mu, logvar)

        # During inference, should return mu
        assert torch.allclose(z, mu)

    def test_vae_model_decode(self, vae_model):
        """Test VAE decoding."""
        z = torch.randn(10, 16)
        reconstruction = vae_model.decode(z)

        assert reconstruction.shape == (10, 50)
        assert torch.all((reconstruction >= 0) & (reconstruction <= 1))

    def test_vae_model_forward(self, vae_model):
        """Test VAE forward pass."""
        x = torch.randn(10, 50)
        reconstruction, mu, logvar = vae_model.forward(x)

        assert reconstruction.shape == (10, 50)
        assert mu.shape == (10, 16)
        assert logvar.shape == (10, 16)

    def test_vae_model_loss_function(self, vae_model):
        """Test VAE loss calculation."""
        x = torch.randn(10, 50)
        reconstruction, mu, logvar = vae_model.forward(x)

        losses = vae_model.loss_function(x, reconstruction, mu, logvar, beta=1.0)

        assert "total_loss" in losses
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert losses["total_loss"].item() > 0

    def test_vae_model_loss_function_beta_vae(self, vae_model):
        """Test VAE loss with beta parameter."""
        x = torch.randn(10, 50)
        reconstruction, mu, logvar = vae_model.forward(x)

        losses_beta1 = vae_model.loss_function(x, reconstruction, mu, logvar, beta=1.0)
        losses_beta2 = vae_model.loss_function(x, reconstruction, mu, logvar, beta=2.0)

        # Higher beta should increase total loss
        assert losses_beta2["total_loss"] >= losses_beta1["total_loss"]

    def test_vae_model_anomaly_score(self, vae_model):
        """Test anomaly score calculation."""
        vae_model.eval()
        x = torch.randn(10, 50)

        total_score, recon_error, kl_div = vae_model.anomaly_score(x)

        assert total_score.shape == (10,)
        assert recon_error.shape == (10,)
        assert kl_div.shape == (10,)
        assert torch.all(total_score >= 0)

    def test_vae_model_without_batch_norm(self):
        """Test VAE model without batch normalization."""
        from ai_engine.anomaly.vae_detector import VAEModel

        model = VAEModel(input_dim=50, latent_dim=16, use_batch_norm=False)
        x = torch.randn(10, 50)
        reconstruction, mu, logvar = model.forward(x)

        assert reconstruction.shape == (10, 50)

    def test_vae_model_with_dropout(self):
        """Test VAE model with dropout."""
        from ai_engine.anomaly.vae_detector import VAEModel

        model = VAEModel(input_dim=50, latent_dim=16, dropout_prob=0.3)
        x = torch.randn(10, 50)
        reconstruction, mu, logvar = model.forward(x)

        assert reconstruction.shape == (10, 50)


class TestVAEAnomalyDetector:
    """Test suite for VAEAnomalyDetector."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        return config

    @pytest.fixture
    def detector(self, mock_config):
        """Create VAE anomaly detector instance."""
        from ai_engine.anomaly.vae_detector import VAEAnomalyDetector

        return VAEAnomalyDetector(mock_config)

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.input_dim == 100
        assert detector.latent_dim == 32
        assert detector.device is not None

    @pytest.mark.asyncio
    async def test_initialize_detector(self, detector):
        """Test detector initialization."""
        await detector.initialize(input_dim=50)

        assert detector.input_dim == 50
        assert detector.model is not None
        assert detector.optimizer is not None
        assert detector.scheduler is not None

    @pytest.mark.asyncio
    async def test_initialize_detector_failure(self, detector):
        """Test detector initialization failure."""
        from ai_engine.core.exceptions import ModelException

        with patch(
            "ai_engine.anomaly.vae_detector.VAEModel",
            side_effect=Exception("Init error"),
        ):
            with pytest.raises(ModelException):
                await detector.initialize(input_dim=50)

    @pytest.mark.asyncio
    async def test_detect_anomaly(self, detector):
        """Test anomaly detection."""
        await detector.initialize(input_dim=50)

        # Set thresholds
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 5.0
        detector.kl_threshold = 5.0

        features = np.random.randn(50)
        result = await detector.detect(features)

        assert result.anomaly_score >= 0
        assert result.reconstruction_error >= 0
        assert result.kl_divergence >= 0
        assert isinstance(result.is_anomaly, bool)
        assert result.confidence >= 0.1
        assert result.confidence <= 0.99

    @pytest.mark.asyncio
    async def test_detect_without_initialization(self, detector):
        """Test detection without initialization."""
        from ai_engine.core.exceptions import AnomalyDetectionException

        features = np.random.randn(50)

        with pytest.raises(AnomalyDetectionException, match="Model not initialized"):
            await detector.detect(features)

    @pytest.mark.asyncio
    async def test_detect_with_context(self, detector):
        """Test detection with context."""
        await detector.initialize(input_dim=50)
        detector.anomaly_threshold = 10.0

        features = np.random.randn(50)
        context = {"protocol_type": "HTTP"}

        result = await detector.detect(features, context=context)

        assert "HTTP" in result.explanation or result.explanation is not None

    @pytest.mark.asyncio
    async def test_detect_failure(self, detector):
        """Test detection failure."""
        from ai_engine.core.exceptions import AnomalyDetectionException

        await detector.initialize(input_dim=50)

        with patch.object(
            detector.model, "anomaly_score", side_effect=Exception("Detection error")
        ):
            features = np.random.randn(50)

            with pytest.raises(AnomalyDetectionException):
                await detector.detect(features)

    @pytest.mark.asyncio
    async def test_train_detector(self, detector):
        """Test detector training."""
        await detector.initialize(input_dim=50)

        training_data = np.random.randn(100, 50)

        history = await detector.train(
            training_data=training_data, num_epochs=2, batch_size=32
        )

        assert "train_loss" in history
        assert "train_recon_loss" in history
        assert "train_kl_loss" in history
        assert len(history["train_loss"]) == 2

    @pytest.mark.asyncio
    async def test_train_with_validation(self, detector):
        """Test training with validation data."""
        await detector.initialize(input_dim=50)

        training_data = np.random.randn(100, 50)
        validation_data = np.random.randn(20, 50)

        history = await detector.train(
            training_data=training_data,
            validation_data=validation_data,
            num_epochs=2,
            batch_size=32,
        )

        assert "val_loss" in history
        assert len(history["val_loss"]) == 2

    @pytest.mark.asyncio
    async def test_train_early_stopping(self, detector):
        """Test training with early stopping."""
        await detector.initialize(input_dim=50)

        training_data = np.random.randn(100, 50)
        validation_data = np.random.randn(20, 50)

        history = await detector.train(
            training_data=training_data,
            validation_data=validation_data,
            num_epochs=100,
            early_stopping_patience=2,
        )

        # Should stop early
        assert len(history["train_loss"]) < 100

    @pytest.mark.asyncio
    async def test_train_without_initialization(self, detector):
        """Test training without initialization."""
        from ai_engine.core.exceptions import AnomalyDetectionException

        training_data = np.random.randn(100, 50)

        with pytest.raises(AnomalyDetectionException, match="Model not initialized"):
            await detector.train(training_data)

    @pytest.mark.asyncio
    async def test_save_model(self, detector, tmp_path):
        """Test model saving."""
        await detector.initialize(input_dim=50)
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 5.0
        detector.kl_threshold = 5.0

        model_path = tmp_path / "vae_model.pt"
        await detector.save_model(str(model_path))

        assert model_path.exists()

    @pytest.mark.asyncio
    async def test_save_model_failure(self, detector, tmp_path):
        """Test model saving failure."""
        from ai_engine.core.exceptions import ModelException

        await detector.initialize(input_dim=50)

        with patch("torch.save", side_effect=Exception("Save error")):
            with pytest.raises(ModelException):
                await detector.save_model(str(tmp_path / "model.pt"))

    @pytest.mark.asyncio
    async def test_load_model(self, detector, tmp_path):
        """Test model loading."""
        await detector.initialize(input_dim=50)

        # Save model first
        model_path = tmp_path / "vae_model.pt"
        await detector.save_model(str(model_path))

        # Create new detector and load
        new_detector = type(detector)(detector.config)
        await new_detector.initialize(input_dim=50)
        await new_detector.load_model(str(model_path))

        assert new_detector.anomaly_threshold is not None

    @pytest.mark.asyncio
    async def test_load_model_failure(self, detector):
        """Test model loading failure."""
        from ai_engine.core.exceptions import ModelException

        await detector.initialize(input_dim=50)

        with pytest.raises(ModelException):
            await detector.load_model("/nonexistent/model.pt")

    def test_preprocess_features_single(self, detector):
        """Test preprocessing single feature vector."""
        detector.feature_mean = torch.zeros(50)
        detector.feature_std = torch.ones(50)

        features = np.random.randn(50)
        processed = detector._preprocess_features(features)

        assert processed.shape == (1, 50)
        assert isinstance(processed, torch.Tensor)

    def test_preprocess_features_batch(self, detector):
        """Test preprocessing batch of features."""
        detector.feature_mean = torch.zeros(50)
        detector.feature_std = torch.ones(50)

        features = np.random.randn(10, 50)
        processed = detector._preprocess_features(features)

        assert processed.shape == (10, 50)

    def test_preprocess_features_without_normalization(self, detector):
        """Test preprocessing without normalization stats."""
        features = np.random.randn(50)
        processed = detector._preprocess_features(features)

        assert processed.shape == (1, 50)

    def test_calculate_normalization_stats(self, detector):
        """Test normalization statistics calculation."""
        training_data = np.random.randn(100, 50)
        detector._calculate_normalization_stats(training_data)

        assert detector.feature_mean is not None
        assert detector.feature_std is not None
        assert detector.feature_mean.shape == (50,)

    def test_create_dataloader(self, detector):
        """Test dataloader creation."""
        detector.feature_mean = torch.zeros(50)
        detector.feature_std = torch.ones(50)

        data = np.random.randn(100, 50)
        dataloader = detector._create_dataloader(data, shuffle=True)

        assert dataloader is not None
        assert len(dataloader) > 0

    @pytest.mark.asyncio
    async def test_calculate_thresholds(self, detector):
        """Test threshold calculation."""
        await detector.initialize(input_dim=50)

        training_data = np.random.randn(100, 50)
        detector._calculate_normalization_stats(training_data)

        await detector._calculate_thresholds(training_data)

        assert detector.anomaly_threshold is not None
        assert detector.reconstruction_threshold is not None
        assert detector.kl_threshold is not None

    def test_is_anomalous_with_thresholds(self, detector):
        """Test anomaly determination with thresholds."""
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 5.0
        detector.kl_threshold = 5.0

        # Normal case
        assert detector._is_anomalous(5.0, 2.0, 2.0) is False

        # Anomalous case
        assert detector._is_anomalous(15.0, 2.0, 2.0) is True
        assert detector._is_anomalous(5.0, 10.0, 2.0) is True
        assert detector._is_anomalous(5.0, 2.0, 10.0) is True

    def test_is_anomalous_without_thresholds(self, detector):
        """Test anomaly determination without thresholds."""
        # Should use fallback heuristic
        assert detector._is_anomalous(15.0, 5.0, 5.0) is True
        assert detector._is_anomalous(5.0, 2.0, 2.0) is False

    def test_calculate_confidence_with_thresholds(self, detector):
        """Test confidence calculation with thresholds."""
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 5.0
        detector.kl_threshold = 5.0

        confidence = detector._calculate_confidence(15.0, 7.0, 7.0)

        assert 0.1 <= confidence <= 0.99

    def test_calculate_confidence_without_thresholds(self, detector):
        """Test confidence calculation without thresholds."""
        confidence = detector._calculate_confidence(10.0, 5.0, 5.0)

        assert confidence == 0.5

    def test_generate_explanation_normal(self, detector):
        """Test explanation generation for normal pattern."""
        detector.anomaly_threshold = 10.0

        explanation = detector._generate_explanation(
            is_anomaly=False, total_score=5.0, recon_error=2.0, kl_div=2.0, context=None
        )

        assert "Normal pattern" in explanation

    def test_generate_explanation_anomaly(self, detector):
        """Test explanation generation for anomaly."""
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 5.0
        detector.kl_threshold = 5.0

        explanation = detector._generate_explanation(
            is_anomaly=True,
            total_score=15.0,
            recon_error=10.0,
            kl_div=10.0,
            context=None,
        )

        assert "Anomaly detected" in explanation

    def test_generate_explanation_with_context(self, detector):
        """Test explanation generation with context."""
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 5.0

        explanation = detector._generate_explanation(
            is_anomaly=True,
            total_score=15.0,
            recon_error=10.0,
            kl_div=2.0,
            context={"protocol_type": "HTTPS"},
        )

        assert "HTTPS" in explanation

    def test_train_epoch(self, detector):
        """Test single training epoch."""
        from torch.utils.data import TensorDataset, DataLoader

        detector.model = Mock()
        detector.model.train = Mock()
        detector.model.return_value = (
            torch.randn(10, 50),
            torch.randn(10, 16),
            torch.randn(10, 16),
        )
        detector.model.loss_function = Mock(
            return_value={
                "total_loss": torch.tensor(1.0),
                "reconstruction_loss": torch.tensor(0.5),
                "kl_loss": torch.tensor(0.5),
            }
        )

        detector.optimizer = Mock()
        detector.scaler = None

        data = torch.randn(100, 50)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        metrics = detector._train_epoch(dataloader)

        assert "total_loss" in metrics
        assert "recon_loss" in metrics
        assert "kl_loss" in metrics

    def test_validate_epoch(self, detector):
        """Test single validation epoch."""
        from torch.utils.data import TensorDataset, DataLoader

        detector.model = Mock()
        detector.model.eval = Mock()
        detector.model.return_value = (
            torch.randn(10, 50),
            torch.randn(10, 16),
            torch.randn(10, 16),
        )
        detector.model.loss_function = Mock(
            return_value={
                "total_loss": torch.tensor(1.0),
                "reconstruction_loss": torch.tensor(0.5),
                "kl_loss": torch.tensor(0.5),
            }
        )

        data = torch.randn(100, 50)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=10)

        metrics = detector._validate_epoch(dataloader)

        assert "total_loss" in metrics
        assert "recon_loss" in metrics
        assert "kl_loss" in metrics


class TestAnomalyResult:
    """Test suite for AnomalyResult dataclass."""

    def test_anomaly_result_creation(self):
        """Test AnomalyResult creation."""
        from ai_engine.anomaly.vae_detector import AnomalyResult

        result = AnomalyResult(
            is_anomaly=True,
            anomaly_score=15.0,
            reconstruction_error=10.0,
            kl_divergence=5.0,
            confidence=0.95,
            explanation="Test anomaly",
            processing_time=0.1,
        )

        assert result.is_anomaly is True
        assert result.anomaly_score == 15.0
        assert result.confidence == 0.95
