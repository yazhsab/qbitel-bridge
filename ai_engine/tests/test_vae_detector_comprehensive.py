"""
Comprehensive Unit Tests for VAE Anomaly Detector
Tests for ai_engine/anomaly/vae_detector.py
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
from pathlib import Path

from ai_engine.anomaly.vae_detector import (
    VAEModel,
    VAEAnomalyDetector,
    AnomalyResult,
)
from ai_engine.core.config import Config
from ai_engine.core.exceptions import AnomalyDetectionException, ModelException


class TestAnomalyResult:
    """Test AnomalyResult dataclass."""
    
    def test_anomaly_result_creation(self):
        """Test creating anomaly result."""
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_score=0.85,
            reconstruction_error=0.6,
            kl_divergence=0.25,
            confidence=0.9,
            explanation="High reconstruction error detected",
            processing_time=0.15
        )
        
        assert result.is_anomaly is True
        assert result.anomaly_score == 0.85
        assert result.reconstruction_error == 0.6
        assert result.kl_divergence == 0.25
        assert result.confidence == 0.9
        assert "reconstruction error" in result.explanation
        assert result.processing_time == 0.15


class TestVAEModel:
    """Test VAEModel class."""
    
    @pytest.fixture
    def vae_model(self):
        """Create VAE model instance."""
        return VAEModel(
            input_dim=100,
            latent_dim=32,
            hidden_dims=[128, 64],
            dropout_prob=0.1,
            use_batch_norm=True
        )
    
    def test_vae_model_initialization(self, vae_model):
        """Test VAE model initialization."""
        assert vae_model.input_dim == 100
        assert vae_model.latent_dim == 32
        assert vae_model.use_batch_norm is True
        assert vae_model.encoder is not None
        assert vae_model.decoder is not None
        assert vae_model.fc_mu is not None
        assert vae_model.fc_logvar is not None
    
    def test_vae_model_default_hidden_dims(self):
        """Test VAE model with default hidden dimensions."""
        model = VAEModel(input_dim=100, latent_dim=32)
        
        assert model.input_dim == 100
        assert model.latent_dim == 32
    
    def test_vae_encode(self, vae_model):
        """Test VAE encoding."""
        x = torch.randn(10, 100)
        
        mu, logvar = vae_model.encode(x)
        
        assert mu.shape == (10, 32)
        assert logvar.shape == (10, 32)
    
    def test_vae_reparameterize_training(self, vae_model):
        """Test reparameterization during training."""
        vae_model.train()
        mu = torch.randn(10, 32)
        logvar = torch.randn(10, 32)
        
        z = vae_model.reparameterize(mu, logvar)
        
        assert z.shape == (10, 32)
    
    def test_vae_reparameterize_inference(self, vae_model):
        """Test reparameterization during inference."""
        vae_model.eval()
        mu = torch.randn(10, 32)
        logvar = torch.randn(10, 32)
        
        z = vae_model.reparameterize(mu, logvar)
        
        # During inference, should return mu
        assert torch.allclose(z, mu)
    
    def test_vae_decode(self, vae_model):
        """Test VAE decoding."""
        z = torch.randn(10, 32)
        
        reconstruction = vae_model.decode(z)
        
        assert reconstruction.shape == (10, 100)
        # Output should be in [0, 1] due to sigmoid
        assert torch.all(reconstruction >= 0)
        assert torch.all(reconstruction <= 1)
    
    def test_vae_forward(self, vae_model):
        """Test VAE forward pass."""
        x = torch.randn(10, 100)
        
        reconstruction, mu, logvar = vae_model.forward(x)
        
        assert reconstruction.shape == (10, 100)
        assert mu.shape == (10, 32)
        assert logvar.shape == (10, 32)
    
    def test_vae_loss_function(self, vae_model):
        """Test VAE loss calculation."""
        x = torch.rand(10, 100)  # Use rand for [0,1] range
        reconstruction, mu, logvar = vae_model.forward(x)
        
        losses = vae_model.loss_function(x, reconstruction, mu, logvar, beta=1.0)
        
        assert "total_loss" in losses
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert losses["total_loss"].item() >= 0
    
    def test_vae_loss_function_with_beta(self, vae_model):
        """Test VAE loss with beta parameter."""
        x = torch.rand(10, 100)
        reconstruction, mu, logvar = vae_model.forward(x)
        
        losses_beta1 = vae_model.loss_function(x, reconstruction, mu, logvar, beta=1.0)
        losses_beta2 = vae_model.loss_function(x, reconstruction, mu, logvar, beta=2.0)
        
        # Higher beta should increase total loss
        assert losses_beta2["total_loss"].item() >= losses_beta1["total_loss"].item()
    
    def test_vae_anomaly_score(self, vae_model):
        """Test anomaly score calculation."""
        vae_model.eval()
        x = torch.rand(10, 100)
        
        total_score, recon_error, kl_div = vae_model.anomaly_score(x)
        
        assert total_score.shape == (10,)
        assert recon_error.shape == (10,)
        assert kl_div.shape == (10,)
        assert torch.all(total_score >= 0)


class TestVAEAnomalyDetector:
    """Test VAEAnomalyDetector class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Mock(spec=Config)
    
    @pytest.fixture
    def detector(self, config):
        """Create VAE anomaly detector instance."""
        return VAEAnomalyDetector(config)
    
    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.config is not None
        assert detector.input_dim == 100
        assert detector.latent_dim == 32
        assert detector.model is None
        assert detector.device is not None
        assert detector.feature_mean is None
        assert detector.feature_std is None
    
    @pytest.mark.asyncio
    async def test_initialize_detector(self, detector):
        """Test detector initialization."""
        await detector.initialize(input_dim=50)
        
        assert detector.input_dim == 50
        assert detector.model is not None
        assert isinstance(detector.model, VAEModel)
        assert detector.optimizer is not None
        assert detector.scheduler is not None
    
    @pytest.mark.asyncio
    async def test_initialize_detector_failure(self, detector):
        """Test detector initialization failure."""
        with patch('ai_engine.anomaly.vae_detector.VAEModel', side_effect=Exception("Init failed")):
            with pytest.raises(ModelException):
                await detector.initialize(input_dim=50)
    
    @pytest.mark.asyncio
    async def test_detect_anomaly(self, detector):
        """Test anomaly detection."""
        await detector.initialize(input_dim=50)
        
        # Set thresholds
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 8.0
        detector.kl_threshold = 2.0
        
        features = np.random.randn(50).astype(np.float32)
        
        result = await detector.detect(features)
        
        assert isinstance(result, AnomalyResult)
        assert isinstance(result.is_anomaly, bool)
        assert result.anomaly_score >= 0
        assert result.reconstruction_error >= 0
        assert result.kl_divergence >= 0
        assert 0 <= result.confidence <= 1
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_detect_without_initialization(self, detector):
        """Test detection without initialization."""
        features = np.random.randn(50).astype(np.float32)
        
        with pytest.raises(AnomalyDetectionException):
            await detector.detect(features)
    
    @pytest.mark.asyncio
    async def test_detect_with_context(self, detector):
        """Test detection with context."""
        await detector.initialize(input_dim=50)
        detector.anomaly_threshold = 10.0
        
        features = np.random.randn(50).astype(np.float32)
        context = {"protocol_type": "http", "source": "network"}
        
        result = await detector.detect(features, context=context)
        
        assert isinstance(result, AnomalyResult)
        assert "http" in result.explanation or "Normal" in result.explanation
    
    @pytest.mark.asyncio
    async def test_train_detector(self, detector):
        """Test detector training."""
        await detector.initialize(input_dim=50)
        
        # Create training data
        training_data = np.random.randn(100, 50).astype(np.float32)
        
        history = await detector.train(
            training_data=training_data,
            num_epochs=2,
            batch_size=32
        )
        
        assert "train_loss" in history
        assert "train_recon_loss" in history
        assert "train_kl_loss" in history
        assert len(history["train_loss"]) > 0
        assert detector.anomaly_threshold is not None
    
    @pytest.mark.asyncio
    async def test_train_with_validation(self, detector):
        """Test training with validation data."""
        await detector.initialize(input_dim=50)
        
        training_data = np.random.randn(100, 50).astype(np.float32)
        validation_data = np.random.randn(20, 50).astype(np.float32)
        
        history = await detector.train(
            training_data=training_data,
            validation_data=validation_data,
            num_epochs=2,
            batch_size=32,
            early_stopping_patience=5
        )
        
        assert "val_loss" in history
        assert len(history["val_loss"]) > 0
    
    @pytest.mark.asyncio
    async def test_train_without_initialization(self, detector):
        """Test training without initialization."""
        training_data = np.random.randn(100, 50).astype(np.float32)
        
        with pytest.raises(AnomalyDetectionException):
            await detector.train(training_data=training_data, num_epochs=2)
    
    @pytest.mark.asyncio
    async def test_save_model(self, detector):
        """Test saving model."""
        await detector.initialize(input_dim=50)
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 8.0
        detector.kl_threshold = 2.0
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            await detector.save_model(model_path)
            
            # Verify file exists
            assert Path(model_path).exists()
        finally:
            Path(model_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_load_model(self, detector):
        """Test loading model."""
        await detector.initialize(input_dim=50)
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 8.0
        detector.kl_threshold = 2.0
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            # Save model
            await detector.save_model(model_path)
            
            # Create new detector and load
            new_detector = VAEAnomalyDetector(detector.config)
            await new_detector.initialize(input_dim=50)
            await new_detector.load_model(model_path)
            
            assert new_detector.anomaly_threshold == 10.0
            assert new_detector.reconstruction_threshold == 8.0
            assert new_detector.kl_threshold == 2.0
        finally:
            Path(model_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self, detector):
        """Test loading model failure."""
        await detector.initialize(input_dim=50)
        
        with pytest.raises(ModelException):
            await detector.load_model("/nonexistent/model.pt")
    
    def test_preprocess_features_single(self, detector):
        """Test preprocessing single feature vector."""
        features = np.random.randn(50).astype(np.float32)
        
        tensor = detector._preprocess_features(features)
        
        assert tensor.shape == (1, 50)
        assert tensor.device == detector.device
    
    def test_preprocess_features_batch(self, detector):
        """Test preprocessing batch of features."""
        features = np.random.randn(10, 50).astype(np.float32)
        
        tensor = detector._preprocess_features(features)
        
        assert tensor.shape == (10, 50)
    
    def test_preprocess_features_with_normalization(self, detector):
        """Test preprocessing with normalization."""
        detector.feature_mean = torch.zeros(50, device=detector.device)
        detector.feature_std = torch.ones(50, device=detector.device)
        
        features = np.random.randn(50).astype(np.float32)
        
        tensor = detector._preprocess_features(features)
        
        assert tensor.shape == (1, 50)
    
    def test_calculate_normalization_stats(self, detector):
        """Test calculating normalization statistics."""
        training_data = np.random.randn(100, 50).astype(np.float32)
        
        detector._calculate_normalization_stats(training_data)
        
        assert detector.feature_mean is not None
        assert detector.feature_std is not None
        assert detector.feature_mean.shape == (50,)
        assert detector.feature_std.shape == (50,)
    
    def test_create_dataloader(self, detector):
        """Test creating data loader."""
        data = np.random.randn(100, 50).astype(np.float32)
        detector.feature_mean = torch.zeros(50)
        detector.feature_std = torch.ones(50)
        
        dataloader = detector._create_dataloader(data, shuffle=True)
        
        assert dataloader is not None
        assert dataloader.batch_size == detector.batch_size
    
    def test_is_anomalous_with_thresholds(self, detector):
        """Test anomaly determination with thresholds."""
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 8.0
        detector.kl_threshold = 2.0
        
        # Normal case
        is_anomaly = detector._is_anomalous(5.0, 4.0, 1.0)
        assert is_anomaly is False
        
        # Anomalous case
        is_anomaly = detector._is_anomalous(15.0, 10.0, 3.0)
        assert is_anomaly is True
    
    def test_is_anomalous_without_thresholds(self, detector):
        """Test anomaly determination without thresholds."""
        # Should use fallback heuristic
        is_anomaly = detector._is_anomalous(15.0, 10.0, 5.0)
        assert is_anomaly is True
        
        is_anomaly = detector._is_anomalous(5.0, 3.0, 2.0)
        assert is_anomaly is False
    
    def test_calculate_confidence(self, detector):
        """Test confidence calculation."""
        detector.anomaly_threshold = 10.0
        detector.reconstruction_threshold = 8.0
        detector.kl_threshold = 2.0
        
        confidence = detector._calculate_confidence(5.0, 4.0, 1.0)
        
        assert 0.1 <= confidence <= 0.99
    
    def test_calculate_confidence_without_thresholds(self, detector):
        """Test confidence calculation without thresholds."""
        confidence = detector._calculate_confidence(5.0, 4.0, 1.0)
        
        assert confidence == 0.5
    
    def test_generate_explanation_normal(self, detector):
        """Test generating explanation for normal pattern."""
        explanation = detector._generate_explanation(
            is_anomaly=False,
            total_score=5.0,
            recon_error=3.0,
            kl_div=2.0,
            context=None
        )
        
        assert "Normal pattern" in explanation
    
    def test_generate_explanation_anomaly(self, detector):
        """Test generating explanation for anomaly."""
        detector.reconstruction_threshold = 5.0
        detector.kl_threshold = 3.0
        
        explanation = detector._generate_explanation(
            is_anomaly=True,
            total_score=15.0,
            recon_error=10.0,
            kl_div=5.0,
            context=None
        )
        
        assert "Anomaly detected" in explanation
    
    def test_generate_explanation_with_context(self, detector):
        """Test generating explanation with context."""
        detector.reconstruction_threshold = 5.0
        
        explanation = detector._generate_explanation(
            is_anomaly=True,
            total_score=15.0,
            recon_error=10.0,
            kl_div=2.0,
            context={"protocol_type": "http"}
        )
        
        assert "http" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
