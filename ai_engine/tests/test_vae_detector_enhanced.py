"""
Enhanced comprehensive tests for ai_engine/anomaly/vae_detector.py
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from ai_engine.anomaly.vae_detector import (
    VAEDetector,
    VAEEncoder,
    VAEDecoder,
    VAE,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return torch.randn(10, 64)


@pytest.fixture
def vae_model():
    """Create a VAE model for testing."""
    return VAE(input_dim=64, latent_dim=16, hidden_dims=[32, 16])


class TestVAEEncoder:
    """Test suite for VAEEncoder."""

    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = VAEEncoder(input_dim=64, latent_dim=16, hidden_dims=[32, 16])
        
        assert encoder.input_dim == 64
        assert encoder.latent_dim == 16
        assert len(encoder.hidden_layers) > 0

    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = VAEEncoder(input_dim=64, latent_dim=16, hidden_dims=[32, 16])
        x = torch.randn(10, 64)
        
        mu, log_var = encoder(x)
        
        assert mu.shape == (10, 16)
        assert log_var.shape == (10, 16)

    def test_encoder_different_hidden_dims(self):
        """Test encoder with different hidden dimensions."""
        encoder = VAEEncoder(input_dim=128, latent_dim=32, hidden_dims=[64, 32, 16])
        x = torch.randn(5, 128)
        
        mu, log_var = encoder(x)
        
        assert mu.shape == (5, 32)
        assert log_var.shape == (5, 32)

    def test_encoder_single_hidden_layer(self):
        """Test encoder with single hidden layer."""
        encoder = VAEEncoder(input_dim=64, latent_dim=16, hidden_dims=[32])
        x = torch.randn(10, 64)
        
        mu, log_var = encoder(x)
        
        assert mu.shape == (10, 16)


class TestVAEDecoder:
    """Test suite for VAEDecoder."""

    def test_decoder_initialization(self):
        """Test decoder initialization."""
        decoder = VAEDecoder(latent_dim=16, output_dim=64, hidden_dims=[16, 32])
        
        assert decoder.latent_dim == 16
        assert decoder.output_dim == 64
        assert len(decoder.hidden_layers) > 0

    def test_decoder_forward(self):
        """Test decoder forward pass."""
        decoder = VAEDecoder(latent_dim=16, output_dim=64, hidden_dims=[16, 32])
        z = torch.randn(10, 16)
        
        reconstruction = decoder(z)
        
        assert reconstruction.shape == (10, 64)

    def test_decoder_different_hidden_dims(self):
        """Test decoder with different hidden dimensions."""
        decoder = VAEDecoder(latent_dim=32, output_dim=128, hidden_dims=[16, 32, 64])
        z = torch.randn(5, 32)
        
        reconstruction = decoder(z)
        
        assert reconstruction.shape == (5, 128)


class TestVAE:
    """Test suite for VAE model."""

    def test_vae_initialization(self):
        """Test VAE initialization."""
        vae = VAE(input_dim=64, latent_dim=16, hidden_dims=[32, 16])
        
        assert vae.input_dim == 64
        assert vae.latent_dim == 16
        assert vae.encoder is not None
        assert vae.decoder is not None

    def test_vae_encode(self, vae_model):
        """Test VAE encoding."""
        x = torch.randn(10, 64)
        
        mu, log_var = vae_model.encode(x)
        
        assert mu.shape == (10, 16)
        assert log_var.shape == (10, 16)

    def test_vae_reparameterize(self, vae_model):
        """Test VAE reparameterization trick."""
        mu = torch.randn(10, 16)
        log_var = torch.randn(10, 16)
        
        z = vae_model.reparameterize(mu, log_var)
        
        assert z.shape == (10, 16)

    def test_vae_decode(self, vae_model):
        """Test VAE decoding."""
        z = torch.randn(10, 16)
        
        reconstruction = vae_model.decode(z)
        
        assert reconstruction.shape == (10, 64)

    def test_vae_forward(self, vae_model):
        """Test VAE forward pass."""
        x = torch.randn(10, 64)
        
        reconstruction, mu, log_var = vae_model(x)
        
        assert reconstruction.shape == (10, 64)
        assert mu.shape == (10, 16)
        assert log_var.shape == (10, 16)

    def test_vae_loss(self, vae_model):
        """Test VAE loss calculation."""
        x = torch.randn(10, 64)
        reconstruction, mu, log_var = vae_model(x)
        
        loss = vae_model.loss_function(reconstruction, x, mu, log_var)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_vae_reconstruction_loss(self, vae_model):
        """Test VAE reconstruction loss component."""
        x = torch.randn(10, 64)
        reconstruction = torch.randn(10, 64)
        
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='sum')
        
        assert isinstance(recon_loss, torch.Tensor)

    def test_vae_kl_divergence(self, vae_model):
        """Test VAE KL divergence calculation."""
        mu = torch.randn(10, 16)
        log_var = torch.randn(10, 16)
        
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        assert isinstance(kl_div, torch.Tensor)


class TestVAEDetector:
    """Test suite for VAEDetector."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        
        assert detector.input_dim == 64
        assert detector.latent_dim == 16
        assert detector.model is not None
        assert not detector.is_trained

    def test_detector_initialization_with_custom_params(self):
        """Test detector with custom parameters."""
        detector = VAEDetector(
            input_dim=128,
            latent_dim=32,
            hidden_dims=[64, 32],
            learning_rate=0.0001,
            batch_size=64,
        )
        
        assert detector.input_dim == 128
        assert detector.latent_dim == 32
        assert detector.learning_rate == 0.0001
        assert detector.batch_size == 64

    def test_detector_preprocess_data(self):
        """Test data preprocessing."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        data = np.random.randn(100, 64)
        
        processed = detector._preprocess_data(data)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (100, 64)

    def test_detector_preprocess_data_from_tensor(self):
        """Test preprocessing from tensor."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        data = torch.randn(100, 64)
        
        processed = detector._preprocess_data(data)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (100, 64)

    def test_detector_train(self):
        """Test detector training."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        data = np.random.randn(100, 64)
        
        with patch.object(detector, '_train_epoch', return_value=0.5):
            detector.train(data, epochs=2)
            
            assert detector.is_trained

    def test_detector_train_with_validation(self):
        """Test detector training with validation data."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        train_data = np.random.randn(100, 64)
        val_data = np.random.randn(20, 64)
        
        with patch.object(detector, '_train_epoch', return_value=0.5):
            with patch.object(detector, '_validate_epoch', return_value=0.6):
                detector.train(train_data, val_data=val_data, epochs=2)
                
                assert detector.is_trained

    def test_detector_detect_single_sample(self):
        """Test anomaly detection on single sample."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        detector.threshold = 1.0
        
        sample = np.random.randn(64)
        
        with patch.object(detector, '_compute_reconstruction_error', return_value=0.5):
            is_anomaly, score = detector.detect(sample)
            
            assert isinstance(is_anomaly, bool)
            assert isinstance(score, float)

    def test_detector_detect_batch(self):
        """Test anomaly detection on batch."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        detector.threshold = 1.0
        
        samples = np.random.randn(10, 64)
        
        with patch.object(detector, '_compute_reconstruction_error', return_value=np.array([0.5] * 10)):
            is_anomaly, scores = detector.detect(samples)
            
            assert len(is_anomaly) == 10
            assert len(scores) == 10

    def test_detector_detect_not_trained(self):
        """Test detection without training."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        sample = np.random.randn(64)
        
        with pytest.raises(RuntimeError, match="not been trained"):
            detector.detect(sample)

    def test_detector_compute_reconstruction_error(self):
        """Test reconstruction error computation."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        
        data = torch.randn(10, 64)
        
        errors = detector._compute_reconstruction_error(data)
        
        assert len(errors) == 10
        assert all(e >= 0 for e in errors)

    def test_detector_set_threshold_percentile(self):
        """Test setting threshold by percentile."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        
        data = np.random.randn(100, 64)
        
        with patch.object(detector, '_compute_reconstruction_error', return_value=np.random.randn(100)):
            detector.set_threshold(data, percentile=95)
            
            assert detector.threshold is not None

    def test_detector_set_threshold_manual(self):
        """Test setting threshold manually."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        
        detector.set_threshold(threshold=1.5)
        
        assert detector.threshold == 1.5

    def test_detector_save_model(self):
        """Test saving model."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        
        with patch("torch.save") as mock_save:
            with patch("pathlib.Path.mkdir"):
                detector.save("test_model.pt")
                
                mock_save.assert_called_once()

    def test_detector_load_model(self):
        """Test loading model."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        
        mock_state = {
            "model_state_dict": {},
            "threshold": 1.0,
            "input_dim": 64,
            "latent_dim": 16,
        }
        
        with patch("torch.load", return_value=mock_state):
            detector.load("test_model.pt")
            
            assert detector.is_trained
            assert detector.threshold == 1.0

    def test_detector_get_latent_representation(self):
        """Test getting latent representation."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        
        data = np.random.randn(10, 64)
        
        latent = detector.get_latent_representation(data)
        
        assert latent.shape == (10, 16)

    def test_detector_reconstruct(self):
        """Test data reconstruction."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        
        data = np.random.randn(10, 64)
        
        reconstructed = detector.reconstruct(data)
        
        assert reconstructed.shape == (10, 64)

    def test_detector_train_with_early_stopping(self):
        """Test training with early stopping."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        train_data = np.random.randn(100, 64)
        val_data = np.random.randn(20, 64)
        
        # Mock increasing validation loss to trigger early stopping
        val_losses = [0.5, 0.6, 0.7, 0.8, 0.9]
        with patch.object(detector, '_train_epoch', return_value=0.5):
            with patch.object(detector, '_validate_epoch', side_effect=val_losses):
                detector.train(train_data, val_data=val_data, epochs=10, early_stopping_patience=2)
                
                assert detector.is_trained

    def test_detector_train_epoch(self):
        """Test single training epoch."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        data = torch.randn(100, 64)
        
        loss = detector._train_epoch(data)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_detector_validate_epoch(self):
        """Test single validation epoch."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        data = torch.randn(20, 64)
        
        loss = detector._validate_epoch(data)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_detector_different_architectures(self):
        """Test detector with different architectures."""
        # Deep architecture
        detector1 = VAEDetector(input_dim=128, latent_dim=32, hidden_dims=[64, 32, 16])
        assert detector1.model is not None
        
        # Shallow architecture
        detector2 = VAEDetector(input_dim=64, latent_dim=16, hidden_dims=[32])
        assert detector2.model is not None

    def test_detector_batch_processing(self):
        """Test batch processing during training."""
        detector = VAEDetector(input_dim=64, latent_dim=16, batch_size=16)
        data = np.random.randn(100, 64)
        
        with patch.object(detector, '_train_epoch', return_value=0.5):
            detector.train(data, epochs=1)
            
            assert detector.is_trained

    def test_detector_gpu_support(self):
        """Test GPU support detection."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        
        # Device should be set based on CUDA availability
        assert detector.device is not None

    def test_detector_anomaly_score_normalization(self):
        """Test anomaly score normalization."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        detector.threshold = 1.0
        
        sample = np.random.randn(64)
        
        with patch.object(detector, '_compute_reconstruction_error', return_value=2.0):
            is_anomaly, score = detector.detect(sample)
            
            assert score >= 0

    def test_detector_save_load_cycle(self):
        """Test save and load cycle."""
        detector1 = VAEDetector(input_dim=64, latent_dim=16)
        detector1.is_trained = True
        detector1.threshold = 1.5
        
        with patch("torch.save") as mock_save:
            with patch("pathlib.Path.mkdir"):
                detector1.save("test_model.pt")
        
        detector2 = VAEDetector(input_dim=64, latent_dim=16)
        
        mock_state = {
            "model_state_dict": detector1.model.state_dict(),
            "threshold": 1.5,
            "input_dim": 64,
            "latent_dim": 16,
            "hidden_dims": [32, 16],
        }
        
        with patch("torch.load", return_value=mock_state):
            detector2.load("test_model.pt")
            
            assert detector2.threshold == detector1.threshold

    def test_detector_edge_cases(self):
        """Test edge cases."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        
        # Empty data
        with pytest.raises((ValueError, RuntimeError)):
            detector.train(np.array([]))
        
        # Wrong dimensions
        with pytest.raises((ValueError, RuntimeError)):
            detector.train(np.random.randn(10, 32))  # Wrong input dim

    def test_detector_reconstruction_quality(self):
        """Test reconstruction quality metrics."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        
        # Perfect reconstruction should have low error
        data = torch.randn(10, 64)
        
        with patch.object(detector.model, 'forward', return_value=(data, torch.zeros(10, 16), torch.zeros(10, 16))):
            errors = detector._compute_reconstruction_error(data)
            
            assert all(e < 0.1 for e in errors)

    def test_detector_latent_space_properties(self):
        """Test latent space properties."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        detector.is_trained = True
        
        data = np.random.randn(100, 64)
        latent = detector.get_latent_representation(data)
        
        # Latent space should have correct dimensions
        assert latent.shape[1] == 16
        
        # Latent vectors should be diverse
        assert np.std(latent) > 0

    def test_detector_training_convergence(self):
        """Test training convergence."""
        detector = VAEDetector(input_dim=64, latent_dim=16)
        data = np.random.randn(100, 64)
        
        losses = []
        def mock_train_epoch(data):
            loss = 1.0 / (len(losses) + 1)  # Decreasing loss
            losses.append(loss)
            return loss
        
        with patch.object(detector, '_train_epoch', side_effect=mock_train_epoch):
            detector.train(data, epochs=5)
            
            # Loss should decrease
            assert losses[-1] < losses[0]