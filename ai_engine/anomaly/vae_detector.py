"""
CRONOS AI Engine - VAE Anomaly Detector

This module implements a Variational Autoencoder (VAE) based anomaly detection
system for identifying unusual patterns in protocol communications.
"""

import logging
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pickle

from ..core.config import Config
from ..core.exceptions import AnomalyDetectionException, ModelException


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    anomaly_score: float
    reconstruction_error: float
    kl_divergence: float
    confidence: float
    explanation: str
    processing_time: float


class VAEModel(nn.Module):
    """
    Variational Autoencoder for anomaly detection.
    
    This model learns to encode normal network traffic patterns into
    a latent space and reconstruct them. Anomalies are identified
    by high reconstruction error and KL divergence.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        dropout_prob: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize VAE model.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
            dropout_prob: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(VAEModel, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            if dropout_prob > 0:
                encoder_layers.append(nn.Dropout(dropout_prob))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            if dropout_prob > 0:
                decoder_layers.append(nn.Dropout(dropout_prob))
            
            prev_dim = hidden_dim
        
        # Final reconstruction layer
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()  # Assuming normalized input [0, 1]
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Use mean during inference for consistency
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate VAE loss (ELBO).
        
        Args:
            x: Input data
            reconstruction: Reconstructed data
            mu: Latent mean
            logvar: Latent log variance
            beta: Beta parameter for beta-VAE
            
        Returns:
            Dictionary containing loss components
        """
        # Reconstruction loss (BCE or MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss (negative ELBO)
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def anomaly_score(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate anomaly scores for input data.
        
        Args:
            x: Input data
            
        Returns:
            Tuple of (total_score, reconstruction_error, kl_divergence)
        """
        self.eval()
        with torch.no_grad():
            reconstruction, mu, logvar = self.forward(x)
            
            # Reconstruction error (per sample)
            recon_error = F.mse_loss(reconstruction, x, reduction='none').sum(dim=1)
            
            # KL divergence (per sample)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Combined anomaly score
            total_score = recon_error + kl_div
            
        return total_score, recon_error, kl_div


class VAEAnomalyDetector:
    """
    VAE-based anomaly detection system.
    
    This class provides a complete anomaly detection system using
    Variational Autoencoders for protocol traffic analysis.
    """
    
    def __init__(self, config: Config):
        """Initialize VAE anomaly detector."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.input_dim = 100  # Will be set based on feature extractor
        self.latent_dim = 32
        self.hidden_dims = [512, 256, 128]
        self.dropout_prob = 0.1
        
        # Training parameters
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.beta = 1.0  # Beta-VAE parameter
        self.weight_decay = 1e-5
        
        # Anomaly detection thresholds
        self.percentile_threshold = 95  # Percentile for threshold calculation
        self.anomaly_threshold = None  # Will be calculated during training
        self.reconstruction_threshold = None
        self.kl_threshold = None
        
        # Model components
        self.model: Optional[VAEModel] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_version = "1.0.0"
        
        # Training state
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Statistics for normalization
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        
        self.logger.info(f"VAEAnomalyDetector initialized with device: {self.device}")
    
    async def initialize(self, input_dim: int) -> None:
        """Initialize the VAE model with specified input dimension."""
        try:
            self.input_dim = input_dim
            
            # Initialize model
            self.model = VAEModel(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                dropout_prob=self.dropout_prob
            ).to(self.device)
            
            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Initialize scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5,
                verbose=True
            )
            
            self.logger.info(f"VAE model initialized with input_dim={input_dim}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VAE model: {e}")
            raise ModelException(f"VAE initialization failed: {e}")
    
    async def detect(
        self,
        features: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> AnomalyResult:
        """
        Detect anomalies in the provided features.
        
        Args:
            features: Input feature vector
            context: Optional context information
            
        Returns:
            Anomaly detection result
        """
        if self.model is None:
            raise AnomalyDetectionException("Model not initialized")
        
        start_time = time.time()
        
        try:
            # Preprocess features
            features_tensor = self._preprocess_features(features)
            
            # Calculate anomaly scores
            total_score, recon_error, kl_div = self.model.anomaly_score(features_tensor)
            
            # Extract scalar values
            total_score = total_score.item()
            recon_error = recon_error.item()
            kl_div = kl_div.item()
            
            # Determine if anomalous
            is_anomaly = self._is_anomalous(total_score, recon_error, kl_div)
            
            # Calculate confidence
            confidence = self._calculate_confidence(total_score, recon_error, kl_div)
            
            # Generate explanation
            explanation = self._generate_explanation(
                is_anomaly, total_score, recon_error, kl_div, context
            )
            
            processing_time = time.time() - start_time
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=total_score,
                reconstruction_error=recon_error,
                kl_divergence=kl_div,
                confidence=confidence,
                explanation=explanation,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            raise AnomalyDetectionException(f"Detection error: {e}")
    
    async def train(
        self,
        training_data: np.ndarray,
        validation_data: Optional[np.ndarray] = None,
        num_epochs: int = 100,
        batch_size: Optional[int] = None,
        early_stopping_patience: int = 15
    ) -> Dict[str, List[float]]:
        """
        Train the VAE model on normal data.
        
        Args:
            training_data: Normal training data
            validation_data: Optional validation data
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        if self.model is None:
            raise AnomalyDetectionException("Model not initialized")
        
        if batch_size is not None:
            self.batch_size = batch_size
        
        self.logger.info(f"Starting VAE training with {len(training_data)} samples")
        
        # Calculate normalization statistics
        self._calculate_normalization_stats(training_data)
        
        # Create data loaders
        train_loader = self._create_dataloader(training_data, shuffle=True)
        val_loader = None
        if validation_data is not None:
            val_loader = self._create_dataloader(validation_data, shuffle=False)
        
        # Training loop
        history = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            history["train_loss"].append(train_metrics["total_loss"])
            history["train_recon_loss"].append(train_metrics["recon_loss"])
            history["train_kl_loss"].append(train_metrics["kl_loss"])
            
            # Validation phase
            if val_loader:
                val_metrics = self._validate_epoch(val_loader)
                
                history["val_loss"].append(val_metrics["total_loss"])
                history["val_recon_loss"].append(val_metrics["recon_loss"])
                history["val_kl_loss"].append(val_metrics["kl_loss"])
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics["total_loss"])
                
                # Early stopping
                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    patience_counter = 0
                    await self._save_best_model()
                else:
                    patience_counter += 1
                
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"train_loss={train_metrics['total_loss']:.4f}, "
                    f"val_loss={val_metrics['total_loss']:.4f}, "
                    f"patience={patience_counter}/{early_stopping_patience}"
                )
                
                if patience_counter >= early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break
            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"train_loss={train_metrics['total_loss']:.4f}"
                )
        
        # Calculate anomaly thresholds from training data
        await self._calculate_thresholds(training_data)
        
        self.logger.info("VAE training completed")
        return history
    
    async def load_model(self, model_path: str) -> None:
        """Load trained VAE model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load metadata
            self.model_version = checkpoint.get('version', '1.0.0')
            self.anomaly_threshold = checkpoint.get('anomaly_threshold')
            self.reconstruction_threshold = checkpoint.get('reconstruction_threshold')
            self.kl_threshold = checkpoint.get('kl_threshold')
            
            # Load normalization statistics
            if 'feature_mean' in checkpoint:
                self.feature_mean = checkpoint['feature_mean'].to(self.device)
            if 'feature_std' in checkpoint:
                self.feature_std = checkpoint['feature_std'].to(self.device)
            
            self.logger.info(f"VAE model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load VAE model: {e}")
            raise ModelException(f"Model loading failed: {e}")
    
    async def save_model(self, model_path: str) -> None:
        """Save trained VAE model."""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'version': self.model_version,
                'anomaly_threshold': self.anomaly_threshold,
                'reconstruction_threshold': self.reconstruction_threshold,
                'kl_threshold': self.kl_threshold,
                'config': {
                    'input_dim': self.input_dim,
                    'latent_dim': self.latent_dim,
                    'hidden_dims': self.hidden_dims,
                }
            }
            
            if self.feature_mean is not None:
                checkpoint['feature_mean'] = self.feature_mean.cpu()
            if self.feature_std is not None:
                checkpoint['feature_std'] = self.feature_std.cpu()
            
            torch.save(checkpoint, model_path)
            self.logger.info(f"VAE model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save VAE model: {e}")
            raise ModelException(f"Model saving failed: {e}")
    
    def _preprocess_features(self, features: np.ndarray) -> torch.Tensor:
        """Preprocess features for model input."""
        # Convert to tensor
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        # Normalize if statistics available
        if self.feature_mean is not None and self.feature_std is not None:
            features_tensor = (features_tensor - self.feature_mean) / (self.feature_std + 1e-8)
        
        return features_tensor
    
    def _calculate_normalization_stats(self, training_data: np.ndarray) -> None:
        """Calculate feature normalization statistics."""
        self.feature_mean = torch.tensor(
            np.mean(training_data, axis=0), dtype=torch.float32, device=self.device
        )
        self.feature_std = torch.tensor(
            np.std(training_data, axis=0), dtype=torch.float32, device=self.device
        )
        
        self.logger.info("Calculated normalization statistics")
    
    def _create_dataloader(self, data: np.ndarray, shuffle: bool = True):
        """Create data loader for training."""
        from torch.utils.data import TensorDataset, DataLoader
        
        # Normalize data
        if self.feature_mean is not None and self.feature_std is not None:
            normalized_data = (data - self.feature_mean.cpu().numpy()) / (
                self.feature_std.cpu().numpy() + 1e-8
            )
        else:
            normalized_data = data
        
        dataset = TensorDataset(
            torch.tensor(normalized_data, dtype=torch.float32)
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def _train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        for batch_data, in train_loader:
            batch_data = batch_data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    reconstruction, mu, logvar = self.model(batch_data)
                    losses = self.model.loss_function(
                        batch_data, reconstruction, mu, logvar, self.beta
                    )
                
                # Backward pass with scaling
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstruction, mu, logvar = self.model(batch_data)
                losses = self.model.loss_function(
                    batch_data, reconstruction, mu, logvar, self.beta
                )
                
                losses['total_loss'].backward()
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total_loss'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
    
    def _validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, in val_loader:
                batch_data = batch_data.to(self.device)
                
                reconstruction, mu, logvar = self.model(batch_data)
                losses = self.model.loss_function(
                    batch_data, reconstruction, mu, logvar, self.beta
                )
                
                total_loss += losses['total_loss'].item()
                total_recon_loss += losses['reconstruction_loss'].item()
                total_kl_loss += losses['kl_loss'].item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
    
    async def _calculate_thresholds(self, training_data: np.ndarray) -> None:
        """Calculate anomaly detection thresholds from training data."""
        self.logger.info("Calculating anomaly detection thresholds")
        
        # Calculate scores for all training data
        scores = []
        recon_errors = []
        kl_divs = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(training_data), self.batch_size):
                batch = training_data[i:i + self.batch_size]
                features_tensor = self._preprocess_features(batch)
                
                total_score, recon_error, kl_div = self.model.anomaly_score(features_tensor)
                
                scores.extend(total_score.cpu().numpy().tolist())
                recon_errors.extend(recon_error.cpu().numpy().tolist())
                kl_divs.extend(kl_div.cpu().numpy().tolist())
        
        # Set thresholds at specified percentile
        self.anomaly_threshold = np.percentile(scores, self.percentile_threshold)
        self.reconstruction_threshold = np.percentile(recon_errors, self.percentile_threshold)
        self.kl_threshold = np.percentile(kl_divs, self.percentile_threshold)
        
        self.logger.info(
            f"Thresholds calculated: anomaly={self.anomaly_threshold:.4f}, "
            f"reconstruction={self.reconstruction_threshold:.4f}, "
            f"kl={self.kl_threshold:.4f}"
        )
    
    def _is_anomalous(
        self, total_score: float, recon_error: float, kl_div: float
    ) -> bool:
        """Determine if the scores indicate an anomaly."""
        if self.anomaly_threshold is None:
            # Fallback to simple heuristic
            return total_score > 10.0
        
        return (
            total_score > self.anomaly_threshold or
            recon_error > self.reconstruction_threshold or
            kl_div > self.kl_threshold
        )
    
    def _calculate_confidence(
        self, total_score: float, recon_error: float, kl_div: float
    ) -> float:
        """Calculate confidence score for the anomaly detection."""
        if self.anomaly_threshold is None:
            return 0.5  # Low confidence without thresholds
        
        # Normalize scores relative to thresholds
        norm_total = min(total_score / self.anomaly_threshold, 5.0)
        norm_recon = min(recon_error / self.reconstruction_threshold, 5.0)
        norm_kl = min(kl_div / self.kl_threshold, 5.0)
        
        # Combine normalized scores
        combined_score = (norm_total + norm_recon + norm_kl) / 3.0
        
        # Convert to confidence (sigmoid-like function)
        confidence = 1.0 / (1.0 + math.exp(-2.0 * (combined_score - 1.0)))
        
        return min(max(confidence, 0.1), 0.99)  # Clamp to reasonable range
    
    def _generate_explanation(
        self,
        is_anomaly: bool,
        total_score: float,
        recon_error: float,
        kl_div: float,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation for the detection result."""
        if not is_anomaly:
            return "Normal pattern detected within expected parameters."
        
        explanations = []
        
        if self.reconstruction_threshold and recon_error > self.reconstruction_threshold:
            ratio = recon_error / self.reconstruction_threshold
            explanations.append(f"High reconstruction error ({ratio:.2f}x threshold)")
        
        if self.kl_threshold and kl_div > self.kl_threshold:
            ratio = kl_div / self.kl_threshold
            explanations.append(f"Unusual latent space distribution ({ratio:.2f}x threshold)")
        
        if not explanations:
            explanations.append("Combined anomaly score exceeds threshold")
        
        base_explanation = "Anomaly detected: " + ", ".join(explanations)
        
        if context and context.get('protocol_type'):
            base_explanation += f" in {context['protocol_type']} traffic"
        
        return base_explanation
    
    async def _save_best_model(self) -> None:
        """Save the current best model."""
        # This would save to a temporary location during training
        # The actual save_model method is used for final model persistence
        pass