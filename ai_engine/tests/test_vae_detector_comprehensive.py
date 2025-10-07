"""
CRONOS AI Engine - VAE Detector Comprehensive Tests

Comprehensive test suite for Variational Autoencoder anomaly detection.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from ai_engine.anomaly.vae_detector import (
    VAEDetector,
    VAEEncoder,
    VAEDecoder,
    VAELoss,
    VAEConfig,
    VAEAnomalyResult,
    VAEException,
)


class TestVAEConfig:
    """Test VAEConfig dataclass."""

    def test_vae_config_creation(self):
        """Test creating VAEConfig instance."""
        config = VAEConfig(
            input_dim=100,
            hidden_dims=[64, 32],
            latent_dim=16,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            beta=1.0,
            reconstruction_loss_weight=1.0,
            kl_loss_weight=1.0,
            anomaly_threshold=0.1,
            device="cpu"
        )
        
        assert config.input_dim == 100
        assert config.hidden_dims == [64, 32]
        assert config.latent_dim == 16
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.beta == 1.0
        assert config.reconstruction_loss_weight == 1.0
        assert config.kl_loss_weight == 1.0
        assert config.anomaly_threshold == 0.1
        assert config.device == "cpu"

    def test_vae_config_defaults(self):
        """Test VAEConfig with default values."""
        config = VAEConfig()
        
        assert config.input_dim == 784
        assert config.hidden_dims == [512, 256]
        assert config.latent_dim == 20
        assert config.learning_rate == 0.001
        assert config.batch_size == 128
        assert config.epochs == 50
        assert config.beta == 1.0
        assert config.device == "cpu"

    def test_vae_config_validation(self):
        """Test VAEConfig validation."""
        # Valid config
        config = VAEConfig(
            input_dim=100,
            hidden_dims=[64, 32],
            latent_dim=16,
            learning_rate=0.001,
            batch_size=32,
            epochs=100
        )
        assert config.is_valid() is True
        
        # Invalid config - negative values
        invalid_config = VAEConfig(
            input_dim=-1,
            hidden_dims=[64, 32],
            latent_dim=16,
            learning_rate=-0.001,
            batch_size=-32,
            epochs=-100
        )
        assert invalid_config.is_valid() is False

    def test_vae_config_device_detection(self):
        """Test VAEConfig device detection."""
        config = VAEConfig(device="auto")
        
        # Should detect available device
        assert config.device in ["cpu", "cuda", "mps"]


class TestVAEEncoder:
    """Test VAEEncoder neural network."""

    @pytest.fixture
    def encoder(self):
        """Create VAEEncoder instance."""
        return VAEEncoder(
            input_dim=100,
            hidden_dims=[64, 32],
            latent_dim=16
        )

    def test_encoder_initialization(self, encoder):
        """Test VAEEncoder initialization."""
        assert encoder.input_dim == 100
        assert encoder.hidden_dims == [64, 32]
        assert encoder.latent_dim == 16
        assert len(encoder.encoder_layers) == 2  # 2 hidden layers
        assert encoder.mu_layer is not None
        assert encoder.logvar_layer is not None

    def test_encoder_forward(self, encoder):
        """Test VAEEncoder forward pass."""
        batch_size = 32
        input_tensor = torch.randn(batch_size, 100)
        
        mu, logvar = encoder(input_tensor)
        
        assert mu.shape == (batch_size, 16)
        assert logvar.shape == (batch_size, 16)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_encoder_reparameterization(self, encoder):
        """Test VAEEncoder reparameterization trick."""
        batch_size = 32
        mu = torch.randn(batch_size, 16)
        logvar = torch.randn(batch_size, 16)
        
        z = encoder.reparameterize(mu, logvar)
        
        assert z.shape == (batch_size, 16)
        assert torch.isfinite(z).all()

    def test_encoder_different_input_sizes(self, encoder):
        """Test VAEEncoder with different input sizes."""
        # Test with different batch sizes
        for batch_size in [1, 16, 64, 128]:
            input_tensor = torch.randn(batch_size, 100)
            mu, logvar = encoder(input_tensor)
            
            assert mu.shape == (batch_size, 16)
            assert logvar.shape == (batch_size, 16)

    def test_encoder_gradient_flow(self, encoder):
        """Test VAEEncoder gradient flow."""
        batch_size = 32
        input_tensor = torch.randn(batch_size, 100, requires_grad=True)
        
        mu, logvar = encoder(input_tensor)
        loss = mu.sum() + logvar.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert torch.isfinite(input_tensor.grad).all()


class TestVAEDecoder:
    """Test VAEDecoder neural network."""

    @pytest.fixture
    def decoder(self):
        """Create VAEDecoder instance."""
        return VAEDecoder(
            latent_dim=16,
            hidden_dims=[32, 64],
            output_dim=100
        )

    def test_decoder_initialization(self, decoder):
        """Test VAEDecoder initialization."""
        assert decoder.latent_dim == 16
        assert decoder.hidden_dims == [32, 64]
        assert decoder.output_dim == 100
        assert len(decoder.decoder_layers) == 2  # 2 hidden layers
        assert decoder.output_layer is not None

    def test_decoder_forward(self, decoder):
        """Test VAEDecoder forward pass."""
        batch_size = 32
        latent_tensor = torch.randn(batch_size, 16)
        
        output = decoder(latent_tensor)
        
        assert output.shape == (batch_size, 100)
        assert torch.isfinite(output).all()

    def test_decoder_different_batch_sizes(self, decoder):
        """Test VAEDecoder with different batch sizes."""
        for batch_size in [1, 16, 64, 128]:
            latent_tensor = torch.randn(batch_size, 16)
            output = decoder(latent_tensor)
            
            assert output.shape == (batch_size, 100)

    def test_decoder_gradient_flow(self, decoder):
        """Test VAEDecoder gradient flow."""
        batch_size = 32
        latent_tensor = torch.randn(batch_size, 16, requires_grad=True)
        
        output = decoder(latent_tensor)
        loss = output.sum()
        loss.backward()
        
        assert latent_tensor.grad is not None
        assert torch.isfinite(latent_tensor.grad).all()


class TestVAELoss:
    """Test VAELoss computation."""

    @pytest.fixture
    def vae_loss(self):
        """Create VAELoss instance."""
        return VAELoss(
            reconstruction_loss_weight=1.0,
            kl_loss_weight=1.0,
            beta=1.0
        )

    def test_vae_loss_initialization(self, vae_loss):
        """Test VAELoss initialization."""
        assert vae_loss.reconstruction_loss_weight == 1.0
        assert vae_loss.kl_loss_weight == 1.0
        assert vae_loss.beta == 1.0

    def test_reconstruction_loss(self, vae_loss):
        """Test reconstruction loss computation."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        reconstructed_data = torch.randn(batch_size, 100)
        
        recon_loss = vae_loss.reconstruction_loss(input_data, reconstructed_data)
        
        assert recon_loss.shape == ()
        assert torch.isfinite(recon_loss).all()
        assert recon_loss >= 0

    def test_kl_divergence_loss(self, vae_loss):
        """Test KL divergence loss computation."""
        batch_size = 32
        mu = torch.randn(batch_size, 16)
        logvar = torch.randn(batch_size, 16)
        
        kl_loss = vae_loss.kl_divergence_loss(mu, logvar)
        
        assert kl_loss.shape == ()
        assert torch.isfinite(kl_loss).all()
        assert kl_loss >= 0

    def test_total_loss(self, vae_loss):
        """Test total VAE loss computation."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        reconstructed_data = torch.randn(batch_size, 100)
        mu = torch.randn(batch_size, 16)
        logvar = torch.randn(batch_size, 16)
        
        total_loss = vae_loss(input_data, reconstructed_data, mu, logvar)
        
        assert total_loss.shape == ()
        assert torch.isfinite(total_loss).all()
        assert total_loss >= 0

    def test_loss_components(self, vae_loss):
        """Test individual loss components."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        reconstructed_data = torch.randn(batch_size, 100)
        mu = torch.randn(batch_size, 16)
        logvar = torch.randn(batch_size, 16)
        
        total_loss, recon_loss, kl_loss = vae_loss.compute_losses(
            input_data, reconstructed_data, mu, logvar
        )
        
        assert total_loss.shape == ()
        assert recon_loss.shape == ()
        assert kl_loss.shape == ()
        assert torch.isfinite(total_loss).all()
        assert torch.isfinite(recon_loss).all()
        assert torch.isfinite(kl_loss).all()

    def test_loss_with_different_weights(self):
        """Test VAE loss with different weight configurations."""
        # Test with different beta values
        for beta in [0.1, 0.5, 1.0, 2.0]:
            vae_loss = VAELoss(beta=beta)
            
            batch_size = 32
            input_data = torch.randn(batch_size, 100)
            reconstructed_data = torch.randn(batch_size, 100)
            mu = torch.randn(batch_size, 16)
            logvar = torch.randn(batch_size, 16)
            
            total_loss = vae_loss(input_data, reconstructed_data, mu, logvar)
            assert torch.isfinite(total_loss).all()


class TestVAEAnomalyResult:
    """Test VAEAnomalyResult dataclass."""

    def test_anomaly_result_creation(self):
        """Test creating VAEAnomalyResult instance."""
        result = VAEAnomalyResult(
            is_anomaly=True,
            anomaly_score=0.85,
            reconstruction_error=0.15,
            kl_divergence=0.05,
            confidence=0.92,
            threshold=0.1,
            metadata={
                "input_shape": (32, 100),
                "latent_dim": 16,
                "model_version": "1.0.0"
            }
        )
        
        assert result.is_anomaly is True
        assert result.anomaly_score == 0.85
        assert result.reconstruction_error == 0.15
        assert result.kl_divergence == 0.05
        assert result.confidence == 0.92
        assert result.threshold == 0.1
        assert result.metadata["input_shape"] == (32, 100)

    def test_anomaly_result_defaults(self):
        """Test VAEAnomalyResult with default values."""
        result = VAEAnomalyResult(
            is_anomaly=False,
            anomaly_score=0.05
        )
        
        assert result.reconstruction_error is None
        assert result.kl_divergence is None
        assert result.confidence is None
        assert result.threshold is None
        assert result.metadata == {}

    def test_anomaly_result_serialization(self):
        """Test VAEAnomalyResult serialization."""
        result = VAEAnomalyResult(
            is_anomaly=True,
            anomaly_score=0.85,
            reconstruction_error=0.15,
            kl_divergence=0.05,
            confidence=0.92,
            threshold=0.1,
            metadata={"model_version": "1.0.0"}
        )
        
        serialized = result.to_dict()
        assert serialized["is_anomaly"] is True
        assert serialized["anomaly_score"] == 0.85
        assert serialized["reconstruction_error"] == 0.15
        assert serialized["metadata"]["model_version"] == "1.0.0"

    def test_anomaly_result_deserialization(self):
        """Test VAEAnomalyResult deserialization."""
        data = {
            "is_anomaly": False,
            "anomaly_score": 0.05,
            "reconstruction_error": 0.02,
            "kl_divergence": 0.01,
            "confidence": 0.88,
            "threshold": 0.1,
            "metadata": {"model_version": "1.0.0"}
        }
        
        result = VAEAnomalyResult.from_dict(data)
        assert result.is_anomaly is False
        assert result.anomaly_score == 0.05
        assert result.reconstruction_error == 0.02
        assert result.metadata["model_version"] == "1.0.0"


class TestVAEDetector:
    """Test VAEDetector main functionality."""

    @pytest.fixture
    def vae_config(self):
        """Create VAE configuration."""
        return VAEConfig(
            input_dim=100,
            hidden_dims=[64, 32],
            latent_dim=16,
            learning_rate=0.001,
            batch_size=32,
            epochs=10,  # Reduced for testing
            beta=1.0,
            anomaly_threshold=0.1,
            device="cpu"
        )

    @pytest.fixture
    def vae_detector(self, vae_config):
        """Create VAEDetector instance."""
        return VAEDetector(vae_config)

    def test_vae_detector_initialization(self, vae_detector, vae_config):
        """Test VAEDetector initialization."""
        assert vae_detector.config == vae_config
        assert vae_detector.encoder is not None
        assert vae_detector.decoder is not None
        assert vae_detector.loss_fn is not None
        assert vae_detector.optimizer is not None
        assert vae_detector.is_trained is False

    def test_vae_detector_forward_pass(self, vae_detector):
        """Test VAEDetector forward pass."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        
        reconstructed, mu, logvar = vae_detector.forward(input_data)
        
        assert reconstructed.shape == (batch_size, 100)
        assert mu.shape == (batch_size, 16)
        assert logvar.shape == (batch_size, 16)
        assert torch.isfinite(reconstructed).all()
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()

    def test_vae_detector_encode(self, vae_detector):
        """Test VAEDetector encoding."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        
        mu, logvar = vae_detector.encode(input_data)
        
        assert mu.shape == (batch_size, 16)
        assert logvar.shape == (batch_size, 16)

    def test_vae_detector_decode(self, vae_detector):
        """Test VAEDetector decoding."""
        batch_size = 32
        latent_data = torch.randn(batch_size, 16)
        
        reconstructed = vae_detector.decode(latent_data)
        
        assert reconstructed.shape == (batch_size, 100)

    def test_vae_detector_sample(self, vae_detector):
        """Test VAEDetector sampling from latent space."""
        batch_size = 32
        
        samples = vae_detector.sample(batch_size)
        
        assert samples.shape == (batch_size, 100)
        assert torch.isfinite(samples).all()

    def test_vae_detector_train_step(self, vae_detector):
        """Test VAEDetector training step."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        
        loss = vae_detector.train_step(input_data)
        
        assert torch.isfinite(loss).all()
        assert loss >= 0

    def test_vae_detector_eval_step(self, vae_detector):
        """Test VAEDetector evaluation step."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        
        loss = vae_detector.eval_step(input_data)
        
        assert torch.isfinite(loss).all()
        assert loss >= 0

    def test_vae_detector_fit(self, vae_detector):
        """Test VAEDetector training."""
        # Create synthetic training data
        train_data = torch.randn(1000, 100)
        
        # Train the model
        vae_detector.fit(train_data)
        
        assert vae_detector.is_trained is True

    def test_vae_detector_predict_anomaly(self, vae_detector):
        """Test VAEDetector anomaly prediction."""
        # Train the model first
        train_data = torch.randn(1000, 100)
        vae_detector.fit(train_data)
        
        # Test with normal data
        normal_data = torch.randn(32, 100)
        result = vae_detector.predict_anomaly(normal_data)
        
        assert isinstance(result, VAEAnomalyResult)
        assert result.is_anomaly is not None
        assert result.anomaly_score is not None
        assert 0 <= result.anomaly_score <= 1

    def test_vae_detector_predict_anomaly_batch(self, vae_detector):
        """Test VAEDetector batch anomaly prediction."""
        # Train the model first
        train_data = torch.randn(1000, 100)
        vae_detector.fit(train_data)
        
        # Test with batch data
        batch_data = torch.randn(64, 100)
        results = vae_detector.predict_anomaly_batch(batch_data)
        
        assert len(results) == 64
        assert all(isinstance(result, VAEAnomalyResult) for result in results)

    def test_vae_detector_compute_reconstruction_error(self, vae_detector):
        """Test VAEDetector reconstruction error computation."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        
        recon_error = vae_detector.compute_reconstruction_error(input_data)
        
        assert recon_error.shape == (batch_size,)
        assert torch.isfinite(recon_error).all()
        assert (recon_error >= 0).all()

    def test_vae_detector_compute_kl_divergence(self, vae_detector):
        """Test VAEDetector KL divergence computation."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        
        kl_div = vae_detector.compute_kl_divergence(input_data)
        
        assert kl_div.shape == (batch_size,)
        assert torch.isfinite(kl_div).all()
        assert (kl_div >= 0).all()

    def test_vae_detector_save_load(self, vae_detector, tmp_path):
        """Test VAEDetector model saving and loading."""
        # Train the model first
        train_data = torch.randn(1000, 100)
        vae_detector.fit(train_data)
        
        # Save the model
        model_path = tmp_path / "vae_model.pt"
        vae_detector.save_model(str(model_path))
        
        # Create new detector and load the model
        new_detector = VAEDetector(vae_detector.config)
        new_detector.load_model(str(model_path))
        
        # Test that loaded model produces same results
        test_data = torch.randn(32, 100)
        original_result = vae_detector.predict_anomaly(test_data)
        loaded_result = new_detector.predict_anomaly(test_data)
        
        assert abs(original_result.anomaly_score - loaded_result.anomaly_score) < 1e-6

    def test_vae_detector_get_model_info(self, vae_detector):
        """Test VAEDetector model information."""
        info = vae_detector.get_model_info()
        
        assert "config" in info
        assert "encoder_params" in info
        assert "decoder_params" in info
        assert "total_params" in info
        assert "is_trained" in info

    def test_vae_detector_set_anomaly_threshold(self, vae_detector):
        """Test VAEDetector anomaly threshold setting."""
        new_threshold = 0.2
        vae_detector.set_anomaly_threshold(new_threshold)
        
        assert vae_detector.config.anomaly_threshold == new_threshold

    def test_vae_detector_error_handling(self, vae_detector):
        """Test VAEDetector error handling."""
        # Test with invalid input
        with pytest.raises(VAEException):
            vae_detector.predict_anomaly(None)
        
        # Test with wrong input shape
        with pytest.raises(VAEException):
            vae_detector.predict_anomaly(torch.randn(32, 50))  # Wrong dimension

    def test_vae_detector_not_trained_error(self, vae_detector):
        """Test VAEDetector error when not trained."""
        test_data = torch.randn(32, 100)
        
        with pytest.raises(VAEException):
            vae_detector.predict_anomaly(test_data)

    def test_vae_detector_different_input_sizes(self, vae_detector):
        """Test VAEDetector with different input sizes."""
        # Train the model first
        train_data = torch.randn(1000, 100)
        vae_detector.fit(train_data)
        
        # Test with different batch sizes
        for batch_size in [1, 16, 64, 128]:
            test_data = torch.randn(batch_size, 100)
            result = vae_detector.predict_anomaly(test_data)
            
            assert isinstance(result, VAEAnomalyResult)
            assert result.anomaly_score is not None

    def test_vae_detector_anomaly_detection_accuracy(self, vae_detector):
        """Test VAEDetector anomaly detection accuracy."""
        # Create normal and anomalous data
        normal_data = torch.randn(1000, 100)
        anomalous_data = torch.randn(100, 100) + 5  # Shifted distribution
        
        # Train on normal data
        vae_detector.fit(normal_data)
        
        # Test anomaly detection
        normal_results = vae_detector.predict_anomaly_batch(normal_data[:100])
        anomalous_results = vae_detector.predict_anomaly_batch(anomalous_data)
        
        # Anomalous data should have higher anomaly scores
        normal_scores = [result.anomaly_score for result in normal_results]
        anomalous_scores = [result.anomaly_score for result in anomalous_results]
        
        assert np.mean(anomalous_scores) > np.mean(normal_scores)

    def test_vae_detector_concurrent_access(self, vae_detector):
        """Test VAEDetector concurrent access."""
        # Train the model first
        train_data = torch.randn(1000, 100)
        vae_detector.fit(train_data)
        
        def predict_anomaly():
            test_data = torch.randn(32, 100)
            return vae_detector.predict_anomaly(test_data)
        
        # Run concurrent operations
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(predict_anomaly) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All operations should complete successfully
        assert len(results) == 10
        assert all(isinstance(result, VAEAnomalyResult) for result in results)

    def test_vae_detector_memory_efficiency(self, vae_detector):
        """Test VAEDetector memory efficiency."""
        # Test with large batch size
        large_batch = torch.randn(1000, 100)
        
        # Should not cause memory issues
        try:
            vae_detector.forward(large_batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Insufficient memory for large batch test")
            else:
                raise

    def test_vae_detector_gradient_flow(self, vae_detector):
        """Test VAEDetector gradient flow during training."""
        batch_size = 32
        input_data = torch.randn(batch_size, 100)
        
        # Perform training step
        loss = vae_detector.train_step(input_data)
        
        # Check that gradients are computed
        for param in vae_detector.encoder.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
        
        for param in vae_detector.decoder.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()

    def test_vae_detector_hyperparameter_sensitivity(self):
        """Test VAEDetector sensitivity to hyperparameters."""
        # Test different beta values
        for beta in [0.1, 0.5, 1.0, 2.0]:
            config = VAEConfig(
                input_dim=100,
                hidden_dims=[64, 32],
                latent_dim=16,
                beta=beta,
                epochs=5  # Reduced for testing
            )
            
            detector = VAEDetector(config)
            train_data = torch.randn(500, 100)
            detector.fit(train_data)
            
            # Should train successfully with different beta values
            assert detector.is_trained is True