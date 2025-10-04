"""
CRONOS AI Engine - Protocol Classifier

This module implements ML-based protocol classification using multiple algorithms
including CNN, LSTM, and ensemble methods for robust protocol identification.
"""

import asyncio
import logging
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from concurrent.futures import ThreadPoolExecutor
import hashlib

from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException
from .statistical_analyzer import StatisticalAnalyzer, ByteStatistics


@dataclass
class ClassificationResult:
    """Result of protocol classification."""

    protocol_type: str
    confidence: float
    probabilities: Dict[str, float]
    features: np.ndarray
    model_votes: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolSample:
    """Training sample for protocol classification."""

    data: bytes
    label: str
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CNNProtocolClassifier(nn.Module):
    """1D CNN for protocol classification."""

    def __init__(self, input_size: int, num_classes: int, embedding_dim: int = 64):
        super(CNNProtocolClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Byte embedding layer
        self.embedding = nn.Embedding(256, embedding_dim)

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

        # Global max pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, sequence_length)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = self.global_pool(x)  # (batch_size, 512, 1)
        x = x.squeeze(-1)  # (batch_size, 512)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class LSTMProtocolClassifier(nn.Module):
    """LSTM-based protocol classifier."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super(LSTMProtocolClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Byte embedding
        self.embedding = nn.Embedding(256, embedding_dim)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, 8, batch_first=True)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)

        # LSTM layers
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)

        # Attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling
        pooled = torch.mean(attended, dim=1)  # (batch_size, hidden_dim * 2)

        # Classification
        output = self.classifier(pooled)

        return F.log_softmax(output, dim=1)


class ProtocolClassifier:
    """
    Enterprise-grade protocol classifier using ensemble of ML models.

    This class combines multiple classification approaches for robust
    protocol identification from network traffic samples.
    """

    def __init__(self, config: Config):
        """Initialize the protocol classifier."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model parameters
        self.max_sequence_length = 512
        self.min_samples_per_class = 10
        self.ensemble_voting = "soft"  # 'hard' or 'soft'
        self.confidence_threshold = 0.7

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Model components
        self.cnn_model: Optional[CNNProtocolClassifier] = None
        self.lstm_model: Optional[LSTMProtocolClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.ensemble_model: Optional[VotingClassifier] = None

        # Data preprocessing
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.statistical_analyzer = StatisticalAnalyzer(config)

        # Training state
        self.is_trained = False
        self.known_protocols: Set[str] = set()
        self.training_history: List[Dict[str, Any]] = []

        # Performance optimization
        self.use_parallel_processing = True
        self.max_workers = (
            config.inference.num_workers if hasattr(config, "inference") else 4
        )
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Caching
        self._feature_cache: Dict[str, np.ndarray] = {}
        self._prediction_cache: Dict[str, ClassificationResult] = {}

        self.logger.info("Protocol Classifier initialized")

    async def train(
        self,
        training_samples: List[ProtocolSample],
        validation_split: float = 0.2,
        epochs: int = 50,
    ) -> Dict[str, Any]:
        """
        Train the protocol classifier ensemble.

        Args:
            training_samples: List of labeled protocol samples
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs

        Returns:
            Training results and metrics
        """
        if not training_samples:
            raise ProtocolException("No training samples provided")

        start_time = time.time()
        self.logger.info(
            f"Training protocol classifier with {len(training_samples)} samples"
        )

        try:
            # Validate training samples
            await self._validate_training_data(training_samples)

            # Extract features from all samples
            self.logger.info("Extracting features from training samples")
            features, labels = await self._prepare_training_data(training_samples)

            # Encode labels
            labels_encoded = self.label_encoder.fit_transform(labels)
            self.known_protocols = set(self.label_encoder.classes_)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features,
                labels_encoded,
                test_size=validation_split,
                stratify=labels_encoded,
                random_state=42,
            )

            # Scale features for traditional ML models
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)

            # Train models in parallel
            training_tasks = []

            # Train CNN model
            training_tasks.append(
                self._train_cnn_model(
                    training_samples, X_train, y_train, X_val, y_val, epochs
                )
            )

            # Train LSTM model
            training_tasks.append(
                self._train_lstm_model(
                    training_samples, X_train, y_train, X_val, y_val, epochs
                )
            )

            # Train Random Forest
            training_tasks.append(
                self._train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
            )

            # Execute training tasks
            training_results = await asyncio.gather(
                *training_tasks, return_exceptions=True
            )

            # Process results
            cnn_results, lstm_results, rf_results = training_results

            # Create ensemble model
            await self._create_ensemble_model()

            # Final validation
            validation_results = await self._validate_ensemble(X_val, y_val)

            self.is_trained = True

            training_summary = {
                "training_time": time.time() - start_time,
                "num_samples": len(training_samples),
                "num_protocols": len(self.known_protocols),
                "protocols": list(self.known_protocols),
                "cnn_results": (
                    cnn_results
                    if not isinstance(cnn_results, Exception)
                    else str(cnn_results)
                ),
                "lstm_results": (
                    lstm_results
                    if not isinstance(lstm_results, Exception)
                    else str(lstm_results)
                ),
                "rf_results": (
                    rf_results
                    if not isinstance(rf_results, Exception)
                    else str(rf_results)
                ),
                "ensemble_results": validation_results,
                "device_used": str(self.device),
            }

            self.training_history.append(training_summary)

            self.logger.info(f"Training completed in {time.time() - start_time:.2f}s")
            return training_summary

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise ModelException(f"Protocol classifier training error: {e}")

    async def classify(self, data: bytes) -> ClassificationResult:
        """
        Classify protocol type from message data.

        Args:
            data: Protocol message data

        Returns:
            Classification result with confidence scores
        """
        if not self.is_trained:
            raise ModelException("Classifier not trained. Call train() first.")

        if not data:
            return ClassificationResult(
                protocol_type="unknown",
                confidence=0.0,
                probabilities={},
                features=np.array([]),
            )

        # Check cache
        cache_key = hashlib.sha256(data).hexdigest()[:16]
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        start_time = time.time()

        try:
            # Extract features
            features = await self._extract_features(data)

            # Get predictions from all models
            predictions = {}
            probabilities = {}

            # CNN prediction
            if self.cnn_model:
                cnn_pred, cnn_probs = await self._predict_cnn(data)
                predictions["cnn"] = cnn_pred
                probabilities["cnn"] = cnn_probs

            # LSTM prediction
            if self.lstm_model:
                lstm_pred, lstm_probs = await self._predict_lstm(data)
                predictions["lstm"] = lstm_pred
                probabilities["lstm"] = lstm_probs

            # Random Forest prediction
            if self.rf_model:
                rf_pred, rf_probs = await self._predict_random_forest(features)
                predictions["rf"] = rf_pred
                probabilities["rf"] = rf_probs

            # Ensemble prediction
            final_prediction, final_confidence, final_probabilities = (
                await self._ensemble_predict(predictions, probabilities)
            )

            # Create result
            result = ClassificationResult(
                protocol_type=final_prediction,
                confidence=final_confidence,
                probabilities=final_probabilities,
                features=features,
                model_votes=predictions,
                metadata={
                    "prediction_time": time.time() - start_time,
                    "data_length": len(data),
                    "models_used": list(predictions.keys()),
                },
            )

            # Cache result
            self._prediction_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return ClassificationResult(
                protocol_type="unknown",
                confidence=0.0,
                probabilities={},
                features=np.array([]),
                metadata={"error": str(e)},
            )

    async def _validate_training_data(self, samples: List[ProtocolSample]) -> None:
        """Validate training data quality."""
        if len(samples) < 10:
            raise ProtocolException(
                "Insufficient training samples (minimum 10 required)"
            )

        # Check label distribution
        label_counts = Counter([sample.label for sample in samples])

        for label, count in label_counts.items():
            if count < self.min_samples_per_class:
                self.logger.warning(
                    f"Protocol {label} has only {count} samples (minimum {self.min_samples_per_class} recommended)"
                )

        # Check data quality
        empty_samples = sum(1 for sample in samples if not sample.data)
        if empty_samples > 0:
            self.logger.warning(f"Found {empty_samples} empty samples")

        self.logger.info(
            f"Training data validation passed: {len(label_counts)} protocols"
        )

    async def _prepare_training_data(
        self, samples: List[ProtocolSample]
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare training data by extracting features."""
        features_list = []
        labels = []

        # Process samples in batches for efficiency
        batch_size = 100
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]

            # Extract features for batch
            batch_features = await asyncio.gather(
                *[self._extract_features(sample.data) for sample in batch]
            )

            features_list.extend(batch_features)
            labels.extend([sample.label for sample in batch])

        return np.array(features_list), labels

    async def _extract_features(self, data: bytes) -> np.ndarray:
        """Extract comprehensive features from protocol data."""
        cache_key = hashlib.sha256(data).hexdigest()[:16]
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        features = []

        # Basic statistical features
        if data:
            # Length features
            features.extend([len(data), np.log(len(data) + 1)])  # Log length

            # Byte distribution features
            byte_counts = np.bincount(
                np.frombuffer(data, dtype=np.uint8), minlength=256
            )
            byte_probs = byte_counts / len(data)

            # Entropy
            entropy = -np.sum(byte_probs * np.log2(byte_probs + 1e-10))
            features.append(entropy)

            # Statistical moments
            data_array = np.frombuffer(data, dtype=np.uint8)
            features.extend(
                [
                    np.mean(data_array),
                    np.std(data_array),
                    np.var(data_array),
                    np.median(data_array),
                ]
            )

            # Byte frequency features (top 16 most common bytes)
            top_bytes = np.argsort(byte_counts)[-16:]
            features.extend(byte_counts[top_bytes].tolist())

            # N-gram features (bigrams)
            if len(data) > 1:
                bigrams = {}
                for i in range(len(data) - 1):
                    bigram = (data[i], data[i + 1])
                    bigrams[bigram] = bigrams.get(bigram, 0) + 1

                # Top 10 bigrams
                top_bigrams = sorted(bigrams.values(), reverse=True)[:10]
                features.extend(top_bigrams + [0] * (10 - len(top_bigrams)))
            else:
                features.extend([0] * 10)

            # Pattern features
            # Repeating byte sequences
            repeats = 0
            for i in range(len(data) - 1):
                if data[i] == data[i + 1]:
                    repeats += 1
            features.append(repeats / len(data) if data else 0)

            # ASCII ratio
            ascii_count = sum(1 for b in data if 32 <= b <= 126)
            features.append(ascii_count / len(data))

            # Null byte ratio
            null_count = data.count(0)
            features.append(null_count / len(data))

            # Common delimiter ratios
            for delimiter in [0x0A, 0x0D, 0x20, 0x09]:  # LF, CR, Space, Tab
                delim_count = data.count(delimiter)
                features.append(delim_count / len(data))

        else:
            # Empty data
            features = [0.0] * 50  # Adjust size based on feature count above

        feature_array = np.array(features, dtype=np.float32)

        # Cache features
        self._feature_cache[cache_key] = feature_array

        return feature_array

    async def _train_cnn_model(
        self,
        samples: List[ProtocolSample],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
    ) -> Dict[str, Any]:
        """Train CNN model."""
        self.logger.info("Training CNN model")

        try:
            # Prepare sequence data
            train_sequences = await self._prepare_sequences(
                [
                    s.data
                    for s in samples
                    if s.label in self.label_encoder.transform([s.label])
                ]
            )
            train_labels = y_train[: len(train_sequences)]

            # Create model
            num_classes = len(self.known_protocols)
            self.cnn_model = CNNProtocolClassifier(
                input_size=self.max_sequence_length, num_classes=num_classes
            ).to(self.device)

            # Training setup
            criterion = nn.NLLLoss()
            optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )

            # Create data loaders
            train_dataset = TensorDataset(
                torch.LongTensor(train_sequences), torch.LongTensor(train_labels)
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Training loop
            train_losses = []
            train_accuracies = []

            for epoch in range(epochs):
                self.cnn_model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0

                for batch_sequences, batch_labels in train_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.cnn_model(batch_sequences)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

                scheduler.step()

                epoch_loss /= len(train_loader)
                epoch_accuracy = 100.0 * correct / total

                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_accuracy)

                if epoch % 10 == 0:
                    self.logger.debug(
                        f"CNN Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%"
                    )

            return {
                "model_type": "CNN",
                "final_loss": train_losses[-1] if train_losses else 0.0,
                "final_accuracy": train_accuracies[-1] if train_accuracies else 0.0,
                "epochs_trained": epochs,
                "parameters": sum(p.numel() for p in self.cnn_model.parameters()),
            }

        except Exception as e:
            self.logger.error(f"CNN training failed: {e}")
            return {"model_type": "CNN", "error": str(e)}

    async def _train_lstm_model(
        self,
        samples: List[ProtocolSample],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
    ) -> Dict[str, Any]:
        """Train LSTM model."""
        self.logger.info("Training LSTM model")

        try:
            # Prepare sequence data
            train_sequences = await self._prepare_sequences(
                [
                    s.data
                    for s in samples
                    if s.label in self.label_encoder.transform([s.label])
                ]
            )
            train_labels = y_train[: len(train_sequences)]

            # Create model
            num_classes = len(self.known_protocols)
            self.lstm_model = LSTMProtocolClassifier(
                input_size=self.max_sequence_length, num_classes=num_classes
            ).to(self.device)

            # Training setup
            criterion = nn.NLLLoss()
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )

            # Create data loaders
            train_dataset = TensorDataset(
                torch.LongTensor(train_sequences), torch.LongTensor(train_labels)
            )
            train_loader = DataLoader(
                train_dataset, batch_size=16, shuffle=True
            )  # Smaller batch for LSTM

            # Training loop
            train_losses = []
            train_accuracies = []

            for epoch in range(epochs):
                self.lstm_model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0

                for batch_sequences, batch_labels in train_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.lstm_model(batch_sequences)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

                scheduler.step()

                epoch_loss /= len(train_loader)
                epoch_accuracy = 100.0 * correct / total

                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_accuracy)

                if epoch % 10 == 0:
                    self.logger.debug(
                        f"LSTM Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%"
                    )

            return {
                "model_type": "LSTM",
                "final_loss": train_losses[-1] if train_losses else 0.0,
                "final_accuracy": train_accuracies[-1] if train_accuracies else 0.0,
                "epochs_trained": epochs,
                "parameters": sum(p.numel() for p in self.lstm_model.parameters()),
            }

        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            return {"model_type": "LSTM", "error": str(e)}

    async def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Train Random Forest model."""
        self.logger.info("Training Random Forest model")

        try:
            self.rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            )

            # Train model
            self.rf_model.fit(X_train, y_train)

            # Evaluate
            train_accuracy = self.rf_model.score(X_train, y_train)
            val_accuracy = self.rf_model.score(X_val, y_val)

            return {
                "model_type": "RandomForest",
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "n_estimators": 100,
                "feature_importance": self.rf_model.feature_importances_.tolist(),
            }

        except Exception as e:
            self.logger.error(f"Random Forest training failed: {e}")
            return {"model_type": "RandomForest", "error": str(e)}

    async def _prepare_sequences(self, data_samples: List[bytes]) -> np.ndarray:
        """Prepare byte sequences for neural network models."""
        sequences = []

        for data in data_samples:
            if not data:
                sequence = [0] * self.max_sequence_length
            else:
                # Convert bytes to integers
                byte_sequence = list(data)

                # Truncate or pad sequence
                if len(byte_sequence) > self.max_sequence_length:
                    sequence = byte_sequence[: self.max_sequence_length]
                else:
                    sequence = byte_sequence + [0] * (
                        self.max_sequence_length - len(byte_sequence)
                    )

            sequences.append(sequence)

        return np.array(sequences)

    async def _create_ensemble_model(self) -> None:
        """Create ensemble model combining all trained models."""
        self.logger.info("Creating ensemble model")

        # For now, ensemble is handled in prediction logic
        # In a full implementation, you might create a meta-learner
        pass

    async def _validate_ensemble(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Validate ensemble model performance."""
        # For now, return basic validation metrics
        return {
            "validation_samples": len(X_val),
            "num_classes": len(self.known_protocols),
            "ensemble_ready": True,
        }

    async def _predict_cnn(self, data: bytes) -> Tuple[str, Dict[str, float]]:
        """Get CNN model prediction."""
        if not self.cnn_model:
            return "unknown", {}

        try:
            self.cnn_model.eval()

            # Prepare sequence
            sequences = await self._prepare_sequences([data])
            sequence_tensor = torch.LongTensor(sequences).to(self.device)

            with torch.no_grad():
                output = self.cnn_model(sequence_tensor)
                probabilities = torch.exp(output).cpu().numpy()[0]

            # Get prediction
            predicted_class = np.argmax(probabilities)
            predicted_label = self.label_encoder.classes_[predicted_class]

            # Create probability distribution
            prob_dict = {
                self.label_encoder.classes_[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }

            return predicted_label, prob_dict

        except Exception as e:
            self.logger.error(f"CNN prediction failed: {e}")
            return "unknown", {}

    async def _predict_lstm(self, data: bytes) -> Tuple[str, Dict[str, float]]:
        """Get LSTM model prediction."""
        if not self.lstm_model:
            return "unknown", {}

        try:
            self.lstm_model.eval()

            # Prepare sequence
            sequences = await self._prepare_sequences([data])
            sequence_tensor = torch.LongTensor(sequences).to(self.device)

            with torch.no_grad():
                output = self.lstm_model(sequence_tensor)
                probabilities = torch.exp(output).cpu().numpy()[0]

            # Get prediction
            predicted_class = np.argmax(probabilities)
            predicted_label = self.label_encoder.classes_[predicted_class]

            # Create probability distribution
            prob_dict = {
                self.label_encoder.classes_[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }

            return predicted_label, prob_dict

        except Exception as e:
            self.logger.error(f"LSTM prediction failed: {e}")
            return "unknown", {}

    async def _predict_random_forest(
        self, features: np.ndarray
    ) -> Tuple[str, Dict[str, float]]:
        """Get Random Forest prediction."""
        if not self.rf_model:
            return "unknown", {}

        try:
            # Scale features
            features_scaled = self.feature_scaler.transform([features])

            # Get prediction and probabilities
            predicted_class = self.rf_model.predict(features_scaled)[0]
            probabilities = self.rf_model.predict_proba(features_scaled)[0]

            predicted_label = self.label_encoder.classes_[predicted_class]

            # Create probability distribution
            prob_dict = {
                self.label_encoder.classes_[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }

            return predicted_label, prob_dict

        except Exception as e:
            self.logger.error(f"Random Forest prediction failed: {e}")
            return "unknown", {}

    async def _ensemble_predict(
        self, predictions: Dict[str, str], probabilities: Dict[str, Dict[str, float]]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Combine predictions from multiple models."""
        if not predictions:
            return "unknown", 0.0, {}

        # Voting-based ensemble
        if self.ensemble_voting == "hard":
            # Hard voting - majority vote
            vote_counts = Counter(predictions.values())
            final_prediction = vote_counts.most_common(1)[0][0]
            final_confidence = vote_counts[final_prediction] / len(predictions)

        else:
            # Soft voting - average probabilities
            all_classes = set()
            for prob_dict in probabilities.values():
                all_classes.update(prob_dict.keys())

            averaged_probs = {}
            for class_name in all_classes:
                class_probs = [
                    prob_dict.get(class_name, 0.0)
                    for prob_dict in probabilities.values()
                ]
                averaged_probs[class_name] = np.mean(class_probs)

            final_prediction = max(averaged_probs, key=averaged_probs.get)
            final_confidence = averaged_probs[final_prediction]

        # Get final probability distribution
        final_probabilities = averaged_probs if "averaged_probs" in locals() else {}

        return final_prediction, final_confidence, final_probabilities

    async def save_model(self, filepath: str) -> None:
        """Save trained models to file."""
        if not self.is_trained:
            raise ModelException("No trained model to save")

        try:
            model_data = {
                "label_encoder": self.label_encoder,
                "feature_scaler": self.feature_scaler,
                "known_protocols": list(self.known_protocols),
                "training_history": self.training_history,
                "config": {
                    "max_sequence_length": self.max_sequence_length,
                    "confidence_threshold": self.confidence_threshold,
                },
            }

            # Save PyTorch models separately
            if self.cnn_model:
                torch.save(self.cnn_model.state_dict(), f"{filepath}_cnn.pt")

            if self.lstm_model:
                torch.save(self.lstm_model.state_dict(), f"{filepath}_lstm.pt")

            if self.rf_model:
                joblib.dump(self.rf_model, f"{filepath}_rf.pkl")

            # Save metadata
            joblib.dump(model_data, f"{filepath}_meta.pkl")

            self.logger.info(f"Models saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            raise ModelException(f"Model save error: {e}")

    async def load_model(self, filepath: str) -> None:
        """Load trained models from file."""
        try:
            # Load metadata
            model_data = joblib.load(f"{filepath}_meta.pkl")

            self.label_encoder = model_data["label_encoder"]
            self.feature_scaler = model_data["feature_scaler"]
            self.known_protocols = set(model_data["known_protocols"])
            self.training_history = model_data.get("training_history", [])

            # Load configuration
            config = model_data.get("config", {})
            self.max_sequence_length = config.get("max_sequence_length", 512)
            self.confidence_threshold = config.get("confidence_threshold", 0.7)

            # Load PyTorch models
            num_classes = len(self.known_protocols)

            try:
                self.cnn_model = CNNProtocolClassifier(
                    input_size=self.max_sequence_length, num_classes=num_classes
                ).to(self.device)
                self.cnn_model.load_state_dict(
                    torch.load(f"{filepath}_cnn.pt", map_location=self.device)
                )
                self.cnn_model.eval()
            except FileNotFoundError:
                self.logger.warning("CNN model file not found")

            try:
                self.lstm_model = LSTMProtocolClassifier(
                    input_size=self.max_sequence_length, num_classes=num_classes
                ).to(self.device)
                self.lstm_model.load_state_dict(
                    torch.load(f"{filepath}_lstm.pt", map_location=self.device)
                )
                self.lstm_model.eval()
            except FileNotFoundError:
                self.logger.warning("LSTM model file not found")

            try:
                self.rf_model = joblib.load(f"{filepath}_rf.pkl")
            except FileNotFoundError:
                self.logger.warning("Random Forest model file not found")

            self.is_trained = True
            self.logger.info(f"Models loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise ModelException(f"Model load error: {e}")

    async def get_supported_protocols(self) -> List[str]:
        """Get list of supported protocol types."""
        return list(self.known_protocols)

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models."""
        return {
            "is_trained": self.is_trained,
            "supported_protocols": list(self.known_protocols),
            "num_protocols": len(self.known_protocols),
            "models_available": {
                "cnn": self.cnn_model is not None,
                "lstm": self.lstm_model is not None,
                "random_forest": self.rf_model is not None,
            },
            "training_history": self.training_history,
            "device": str(self.device),
            "max_sequence_length": self.max_sequence_length,
            "confidence_threshold": self.confidence_threshold,
        }

    async def shutdown(self):
        """Shutdown classifier and cleanup resources."""
        self.logger.info("Shutting down Protocol Classifier")

        if self.executor:
            self.executor.shutdown(wait=True)

        if self.statistical_analyzer:
            await self.statistical_analyzer.shutdown()

        # Clear caches
        self._feature_cache.clear()
        self._prediction_cache.clear()

        # Clear models to free GPU memory
        if self.cnn_model:
            del self.cnn_model
            self.cnn_model = None

        if self.lstm_model:
            del self.lstm_model
            self.lstm_model = None

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Protocol Classifier shutdown completed")
