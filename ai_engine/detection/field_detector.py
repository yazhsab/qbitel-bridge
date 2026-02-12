"""
QBITEL Engine - Field Detector

This module implements BiLSTM-CRF based field detection for automatic
identification of field boundaries and types in protocol messages.
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import os

CRF = None  # type: ignore[assignment]
try:  # Optional dependency for CRF layer
    from torchcrf import CRF  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    try:
        from TorchCRF import CRF  # type: ignore
    except ImportError:
        CRF = None  # type: ignore[assignment]
    else:
        CRF = CRF
else:
    CRF = CRF
import pickle

from ..core.config import Config
from ..core.exceptions import FieldDetectionException, ModelException


class FieldType(str, Enum):
    """Field data types."""

    INTEGER = "integer"
    STRING = "string"
    BINARY = "binary"
    TIMESTAMP = "timestamp"
    ADDRESS = "address"
    LENGTH = "length"
    CHECKSUM = "checksum"
    DELIMITER = "delimiter"
    RESERVED = "reserved"
    UNKNOWN = "unknown"


class IOBTag(str, Enum):
    """IOB tagging scheme for sequence labeling."""

    OUTSIDE = "O"
    BEGIN_FIELD = "B-FIELD"
    INSIDE_FIELD = "I-FIELD"


@dataclass
class FieldBoundary:
    """Represents a detected field boundary."""

    start_pos: int
    end_pos: int
    field_type: FieldType
    confidence: float
    field_name: Optional[str] = None
    raw_value: Optional[bytes] = None
    parsed_value: Optional[Any] = None


@dataclass
class FieldPrediction:
    """Complete field detection prediction."""

    message_length: int
    detected_fields: List[FieldBoundary]
    confidence_score: float
    processing_time: float
    model_version: str


class FieldDetectionDataset(Dataset):
    """Dataset wrapper preparing field detection training samples."""

    def __init__(
        self,
        samples: List[Tuple[bytes, List[Tuple[int, int, str]]]],
        max_sequence_length: int,
        tag_to_id: Dict[str, int],
    ) -> None:
        self.max_sequence_length = max_sequence_length
        self.tag_to_id = tag_to_id
        self._encoded_samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
            self._encode_sample(message, annotations or []) for message, annotations in samples
        ]

    def __len__(self) -> int:
        return len(self._encoded_samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._encoded_samples[index]

    def _encode_sample(
        self, message: bytes, annotations: List[Tuple[int, int, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        message_bytes = bytes(message)
        truncated = message_bytes[: self.max_sequence_length]

        sequence = list(truncated)
        seq_length = len(sequence)

        mask = [True] * seq_length
        tags = [IOBTag.OUTSIDE.value] * seq_length

        for start, end, _ in annotations:
            if end <= start:
                continue
            start_idx = max(0, int(start))
            end_idx = min(int(end), self.max_sequence_length, len(message_bytes))
            if start_idx >= end_idx or start_idx >= self.max_sequence_length:
                continue

            # Adjust end index if truncated
            end_idx = min(end_idx, seq_length)
            if end_idx <= start_idx:
                continue

            tags[start_idx] = IOBTag.BEGIN_FIELD.value
            for idx in range(start_idx + 1, end_idx):
                if idx >= len(tags):
                    break
                tags[idx] = IOBTag.INSIDE_FIELD.value

        pad_length = self.max_sequence_length - seq_length
        if pad_length > 0:
            sequence.extend([0] * pad_length)
            tags.extend([IOBTag.OUTSIDE.value] * pad_length)
            mask.extend([False] * pad_length)

        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        tags_tensor = torch.tensor([self.tag_to_id[tag] for tag in tags], dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        return sequence_tensor, tags_tensor, mask_tensor


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF model for field boundary detection.

    This model combines bidirectional LSTM for sequence modeling
    with CRF for structured prediction of field boundaries.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_tags: int = 3,  # O, B-FIELD, I-FIELD
        dropout: float = 0.1,
        use_char_embeddings: bool = True,
        char_vocab_size: int = 256,
    ):
        """Initialize BiLSTM-CRF model."""
        super(BiLSTMCRF, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tags = num_tags
        self.use_char_embeddings = use_char_embeddings

        # Word embeddings (for byte tokens)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Character-level embeddings (for individual bytes)
        if use_char_embeddings:
            self.char_embeddings = nn.Embedding(char_vocab_size, 32)
            self.char_lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
            total_embedding_dim = embedding_dim + 64  # 64 from bidirectional char LSTM
        else:
            total_embedding_dim = embedding_dim

        # BiLSTM layers
        self.lstm = nn.LSTM(
            total_embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Linear layer to map LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        # CRF layer with backward compatibility handling
        try:
            self.crf = CRF(num_tags, batch_first=True)
        except TypeError:  # Older torchcrf versions do not support batch_first
            self.crf = CRF(num_tags)

    def _get_char_features(self, sequences: torch.Tensor) -> torch.Tensor:
        """Extract character-level features."""
        batch_size, seq_len = sequences.shape

        # Convert byte tokens to character sequences (simplified)
        char_features = []

        for b in range(batch_size):
            seq_char_features = []
            for s in range(seq_len):
                # Simple character representation (byte value as character)
                byte_val = sequences[b, s].item()
                char_emb = self.char_embeddings(torch.tensor([byte_val % 256], device=sequences.device))
                char_out, _ = self.char_lstm(char_emb.unsqueeze(0))
                char_feat = char_out.squeeze(0)[-1]  # Take last hidden state
                seq_char_features.append(char_feat)

            char_features.append(torch.stack(seq_char_features))

        return torch.stack(char_features)

    def _get_lstm_features(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Get LSTM features for sequences."""
        batch_size, seq_len = sequences.shape

        # Word embeddings
        word_embeds = self.word_embeddings(sequences)

        # Character embeddings (if enabled)
        if self.use_char_embeddings:
            char_features = self._get_char_features(sequences)
            # Concatenate word and character embeddings
            embeddings = torch.cat([word_embeds, char_features], dim=-1)
        else:
            embeddings = word_embeds

        # Apply dropout to embeddings
        embeddings = self.dropout(embeddings)

        # Pack sequences for LSTM
        lengths = masks.sum(dim=1).cpu()
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        # BiLSTM
        packed_lstm_out, _ = self.lstm(packed_embeddings)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)

        # Apply dropout to LSTM output
        lstm_out = self.dropout(lstm_out)

        # Map to tag space
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def forward(self, sequences: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        lstm_feats = self._get_lstm_features(sequences, masks)
        log_likelihood = self.crf(lstm_feats, tags, mask=masks.bool())
        if log_likelihood.dim() > 0:
            log_likelihood = log_likelihood.mean()
        return -log_likelihood

    def predict(self, sequences: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        """Predict tag sequences."""
        lstm_feats = self._get_lstm_features(sequences, masks)
        if hasattr(self.crf, "decode"):
            return self.crf.decode(lstm_feats, mask=masks.bool())  # type: ignore[attr-defined]
        return self.crf.viterbi_decode(lstm_feats, masks.bool())


class FieldDetector:
    """
    Main field detection system using BiLSTM-CRF.

    This class provides high-level interface for field detection,
    including training, inference, and post-processing.
    """

    def __init__(self, config: Config):
        """Initialize field detector."""
        if CRF is None:
            raise FieldDetectionException("torchcrf is required for FieldDetector. Install with 'pip install torchcrf'.")

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model parameters
        self.vocab_size = 256  # Byte values
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.num_layers = 2
        self.max_sequence_length = 512

        # Tag mappings
        self.tag_to_id = {tag.value: idx for idx, tag in enumerate(IOBTag)}
        self.id_to_tag = {idx: tag for tag, idx in self.tag_to_id.items()}

        # Field type classifier (separate model for type inference)
        self.type_classifier = None

        # Model components
        self.model: Optional[BiLSTMCRF] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_version = "1.0.0"

        # Training state
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        self.logger.info(f"FieldDetector initialized with device: {self.device}")

    async def initialize(self) -> None:
        """Initialize the field detector model."""
        try:
            # Initialize model
            self.model = BiLSTMCRF(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_tags=len(IOBTag),
            ).to(self.device)

            # Initialize type classifier
            await self._initialize_type_classifier()

            self.logger.info("FieldDetector model initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize FieldDetector: {e}")
            raise ModelException(f"FieldDetector initialization failed: {e}")

    async def detect_boundaries(self, message_data: bytes, protocol_type: Optional[str] = None) -> List[FieldBoundary]:
        """
        Detect field boundaries in a protocol message.

        Args:
            message_data: Raw protocol message data
            protocol_type: Optional protocol type hint

        Returns:
            List of detected field boundaries
        """
        if self.model is None:
            raise FieldDetectionException("Model not initialized")

        try:
            # Preprocess message
            sequences, masks = self._preprocess_message(message_data)

            # Run inference
            with torch.no_grad():
                self.model.eval()
                predictions = self.model.predict(sequences, masks)

            # Post-process predictions to field boundaries
            field_boundaries = self._postprocess_predictions(predictions[0], message_data, protocol_type)

            return field_boundaries

        except Exception as e:
            self.logger.error(f"Field boundary detection failed: {e}")
            raise FieldDetectionException(f"Boundary detection error: {e}")

    async def infer_field_types(self, field_boundaries: List[FieldBoundary]) -> List[FieldBoundary]:
        """Infer field types for detected boundaries."""
        # Check if ML classifier is enabled via feature flag
        ml_classifier_enabled = getattr(self.config, "field_detection_ml_classifier_enabled", False)

        if not self.type_classifier or not ml_classifier_enabled:
            # Use simple heuristic-based type inference
            if not ml_classifier_enabled:
                self.logger.info(
                    "ML-based field type classifier is disabled (scaffold code). "
                    "Using heuristic-based inference. Enable with field_detection_ml_classifier_enabled=true"
                )
            return self._heuristic_type_inference(field_boundaries)

        # Use ML-based type inference
        for boundary in field_boundaries:
            if boundary.raw_value:
                predicted_type = await self._predict_field_type(boundary.raw_value)
                boundary.field_type = predicted_type

        return field_boundaries

    def train(
        self,
        training_data: List[Tuple[bytes, List[Tuple[int, int, str]]]],
        validation_data: Optional[List[Tuple[bytes, List[Tuple[int, int, str]]]]] = None,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        early_stopping_patience: int = 10,
        save_best_only: bool = True,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the field detection model with comprehensive metrics and checkpointing.

        Args:
            training_data: List of (message, field_annotations) pairs
            validation_data: Optional validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            early_stopping_patience: Epochs to wait before early stopping
            save_best_only: Only save model when validation improves
            checkpoint_dir: Directory for saving checkpoints

        Returns:
            Training history with comprehensive metrics
        """
        if self.model is None:
            raise FieldDetectionException("Model not initialized")

        self.logger.info(f"Starting training with {len(training_data)} samples")

        # Prepare data loaders
        train_loader = self._create_data_loader(training_data, batch_size, shuffle=True)
        val_loader = None
        if validation_data:
            val_loader = self._create_data_loader(validation_data, batch_size, shuffle=False)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)

        # Training loop with comprehensive metrics
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "val_accuracy": [],
            "learning_rate": [],
            "epoch_time": [],
        }

        best_val_f1 = 0.0
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Record learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(current_lr)

            # Validation phase with comprehensive metrics
            if val_loader:
                val_metrics = self._validate_epoch_comprehensive(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_f1"].append(val_metrics["f1"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_accuracy"].append(val_metrics["accuracy"])

                # Learning rate scheduling
                self.scheduler.step(val_metrics["loss"])

                # Early stopping and model saving
                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    epochs_without_improvement = 0

                    if save_best_only:
                        checkpoint_path = self._save_checkpoint(epoch, val_metrics, checkpoint_dir)
                        self.logger.info(f"Saved best model checkpoint: {checkpoint_path}")
                else:
                    epochs_without_improvement += 1

                epoch_time = time.time() - epoch_start
                history["epoch_time"].append(epoch_time)

                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s): "
                    f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                    f"val_f1={val_metrics['f1']:.4f}, val_precision={val_metrics['precision']:.4f}, "
                    f"val_recall={val_metrics['recall']:.4f}, val_accuracy={val_metrics['accuracy']:.4f}, "
                    f"lr={current_lr:.6f}"
                )

                # Early stopping check
                if epochs_without_improvement >= early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch+1} epochs "
                        f"({early_stopping_patience} epochs without improvement)"
                    )
                    break
            else:
                epoch_time = time.time() - epoch_start
                history["epoch_time"].append(epoch_time)
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s): " f"train_loss={train_loss:.4f}, lr={current_lr:.6f}"
                )

        # Final summary
        total_time = sum(history["epoch_time"])
        self.logger.info(f"Training completed in {total_time:.2f}s. " f"Best validation F1: {best_val_f1:.4f}")

        # Add summary statistics
        history["summary"] = {
            "best_val_f1": best_val_f1,
            "total_epochs": len(history["train_loss"]),
            "total_time_seconds": total_time,
            "final_learning_rate": (history["learning_rate"][-1] if history["learning_rate"] else learning_rate),
        }

        return history

    def train_batch(
        self,
        training_batches: List[List[Tuple[bytes, List[Tuple[int, int, str]]]]],
        validation_data: Optional[List[Tuple[bytes, List[Tuple[int, int, str]]]]] = None,
        num_epochs_per_batch: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Train model on multiple batches sequentially (incremental learning).

        Args:
            training_batches: List of training data batches
            validation_data: Optional validation data
            num_epochs_per_batch: Epochs to train on each batch
            learning_rate: Learning rate
            batch_size: Batch size

        Returns:
            Combined training history across all batches
        """
        if self.model is None:
            raise FieldDetectionException("Model not initialized")

        self.logger.info(f"Starting batch training with {len(training_batches)} batches")

        combined_history = {
            "batch_histories": [],
            "batch_metrics": [],
            "overall_train_loss": [],
            "overall_val_f1": [],
        }

        for batch_idx, batch_data in enumerate(training_batches):
            self.logger.info(f"Training on batch {batch_idx + 1}/{len(training_batches)} " f"({len(batch_data)} samples)")

            # Train on this batch
            batch_history = self.train(
                training_data=batch_data,
                validation_data=validation_data,
                num_epochs=num_epochs_per_batch,
                learning_rate=learning_rate,
                batch_size=batch_size,
                early_stopping_patience=5,
                save_best_only=True,
            )

            combined_history["batch_histories"].append(batch_history)
            combined_history["overall_train_loss"].extend(batch_history["train_loss"])
            if "val_f1" in batch_history and batch_history["val_f1"]:
                combined_history["overall_val_f1"].extend(batch_history["val_f1"])

            # Record batch metrics
            batch_metrics = {
                "batch_idx": batch_idx,
                "samples": len(batch_data),
                "final_train_loss": batch_history["train_loss"][-1],
                "best_val_f1": batch_history["summary"]["best_val_f1"],
            }
            combined_history["batch_metrics"].append(batch_metrics)

            self.logger.info(
                f"Batch {batch_idx + 1} completed: "
                f"final_loss={batch_metrics['final_train_loss']:.4f}, "
                f"best_f1={batch_metrics['best_val_f1']:.4f}"
            )

        self.logger.info("Batch training completed")
        return combined_history

    async def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model_version = checkpoint.get("version", "1.0.0")

            if "type_classifier" in checkpoint:
                self.type_classifier = checkpoint["type_classifier"]

            self.logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelException(f"Model loading failed: {e}")

    async def save_model(self, model_path: str) -> None:
        """Save the trained model asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_model, model_path)

    def _preprocess_message(self, message_data: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess message data for model input."""
        # Convert bytes to sequence of integers
        byte_sequence = list(message_data)

        # Truncate or pad to max_sequence_length
        if len(byte_sequence) > self.max_sequence_length:
            byte_sequence = byte_sequence[: self.max_sequence_length]

        # Create padding mask
        actual_length = len(byte_sequence)
        mask = [1] * actual_length + [0] * (self.max_sequence_length - actual_length)

        # Pad sequence
        byte_sequence.extend([0] * (self.max_sequence_length - actual_length))

        # Convert to tensors
        sequences = torch.tensor([byte_sequence], dtype=torch.long, device=self.device)
        masks = torch.tensor([mask], dtype=torch.float, device=self.device)

        return sequences, masks

    def _postprocess_predictions(
        self, predictions: List[int], message_data: bytes, protocol_type: Optional[str]
    ) -> List[FieldBoundary]:
        """Post-process model predictions to field boundaries."""
        field_boundaries = []
        current_field_start = None

        for i, tag_id in enumerate(predictions):
            if i >= len(message_data):
                break

            tag = self.id_to_tag[tag_id]

            if tag == IOBTag.BEGIN_FIELD.value:
                # Start of new field
                if current_field_start is not None:
                    # End previous field
                    field_boundaries.append(
                        FieldBoundary(
                            start_pos=current_field_start,
                            end_pos=i,
                            field_type=FieldType.UNKNOWN,
                            confidence=0.8,  # Default confidence
                            raw_value=message_data[current_field_start:i],
                        )
                    )
                current_field_start = i

            elif tag == IOBTag.OUTSIDE.value:
                # End of field
                if current_field_start is not None:
                    field_boundaries.append(
                        FieldBoundary(
                            start_pos=current_field_start,
                            end_pos=i,
                            field_type=FieldType.UNKNOWN,
                            confidence=0.8,
                            raw_value=message_data[current_field_start:i],
                        )
                    )
                    current_field_start = None

            # IOBTag.INSIDE_FIELD continues current field

        # Handle field that extends to end of message
        if current_field_start is not None:
            field_boundaries.append(
                FieldBoundary(
                    start_pos=current_field_start,
                    end_pos=len(message_data),
                    field_type=FieldType.UNKNOWN,
                    confidence=0.8,
                    raw_value=message_data[current_field_start:],
                )
            )

        return field_boundaries

    def _heuristic_type_inference(self, field_boundaries: List[FieldBoundary]) -> List[FieldBoundary]:
        """Simple heuristic-based field type inference."""
        for boundary in field_boundaries:
            if not boundary.raw_value:
                continue

            field_data = boundary.raw_value
            field_length = len(field_data)

            # Heuristic rules
            if field_length == 1:
                boundary.field_type = FieldType.INTEGER
            elif field_length == 2:
                boundary.field_type = FieldType.LENGTH
            elif field_length == 4:
                # Could be timestamp, address, or length
                boundary.field_type = FieldType.ADDRESS
            elif field_length == 8:
                boundary.field_type = FieldType.TIMESTAMP
            elif all(32 <= b <= 126 for b in field_data):
                # Printable ASCII
                boundary.field_type = FieldType.STRING
            elif field_data == b"\x00" * field_length:
                boundary.field_type = FieldType.RESERVED
            else:
                boundary.field_type = FieldType.BINARY

        return field_boundaries

    async def _predict_field_type(self, field_data: bytes) -> FieldType:
        """
        Predict field type using ML classifier.

        NOTE: This is scaffold code. ML-based classifier not yet implemented.
        Falls back to heuristic-based inference.
        """
        self.logger.warning(
            "ML-based field type prediction called but not implemented (scaffold code). " "Using heuristic fallback."
        )

        field_length = len(field_data)

        if field_length == 4 and all(b < 128 for b in field_data):
            return FieldType.INTEGER
        elif all(32 <= b <= 126 for b in field_data):
            return FieldType.STRING
        else:
            return FieldType.BINARY

    async def _initialize_type_classifier(self) -> None:
        """
        Initialize field type classifier.

        NOTE: This is scaffold code. ML-based classifier not yet implemented.
        """
        self.type_classifier = None
        ml_classifier_enabled = getattr(self.config, "field_detection_ml_classifier_enabled", False)

        if ml_classifier_enabled:
            self.logger.warning(
                "ML-based field type classifier requested but not implemented (scaffold code). "
                "Falling back to heuristic-based inference. "
                "Set field_detection_ml_classifier_enabled=false to suppress this warning."
            )
        else:
            self.logger.info("Type classifier initialized (heuristic-based mode)")

    def _create_data_loader(
        self,
        data: List[Tuple[bytes, List[Tuple[int, int, str]]]],
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        Create data loader for training/validation.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        dataset = FieldDetectionDataset(
            samples=data,
            max_sequence_length=self.max_sequence_length,
            tag_to_id=self.tag_to_id,
        )

        if len(dataset) == 0:
            self.logger.warning("Field detection dataset is empty; loader will be empty")

        return DataLoader(
            dataset,
            batch_size=(min(batch_size, max(1, len(dataset))) if len(dataset) > 0 else batch_size),
            shuffle=shuffle and len(dataset) > 1,
            drop_last=False,
        )

    def _train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        if self.model is None or self.optimizer is None:
            raise FieldDetectionException("Model and optimizer must be initialized before training")

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for sequences, tags, masks in train_loader:
            sequences = sequences.to(self.device)
            tags = tags.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            loss = self.model(sequences, tags, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, val_loader) -> Tuple[float, float]:
        """Validate for one epoch."""
        if self.model is None:
            raise FieldDetectionException("Model must be initialized before validation")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        with torch.no_grad():
            for sequences, tags, masks in val_loader:
                sequences = sequences.to(self.device)
                tags = tags.to(self.device)
                masks = masks.to(self.device)

                batch_loss = self.model(sequences, tags, masks)
                total_loss += batch_loss.item()
                num_batches += 1

                predictions = self.model.predict(sequences, masks)
                tags_cpu = tags.cpu().numpy()
                masks_cpu = masks.bool().cpu().numpy()

                outside_tag_id = self.tag_to_id[IOBTag.OUTSIDE.value]

                for pred_seq, true_seq, mask_seq in zip(predictions, tags_cpu, masks_cpu):
                    for pred_label, true_label, mask_flag in zip(pred_seq, true_seq, mask_seq):
                        if not mask_flag:
                            continue

                        is_true_positive = true_label != outside_tag_id
                        is_pred_positive = pred_label != outside_tag_id

                        if is_true_positive and is_pred_positive:
                            true_positive += 1
                        elif is_true_positive and not is_pred_positive:
                            false_negative += 1
                        elif not is_true_positive and is_pred_positive:
                            false_positive += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = true_positive / precision_denominator if precision_denominator > 0 else 0.0
        recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0

        if precision == 0.0 and recall == 0.0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

    def _validate_epoch_comprehensive(self, val_loader) -> Dict[str, float]:
        """
        Validate for one epoch with comprehensive metrics.

        Returns:
            Dictionary with loss, f1, precision, recall, accuracy, and per-class metrics
        """
        if self.model is None:
            raise FieldDetectionException("Model must be initialized before validation")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Confusion matrix components
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        # Per-tag metrics
        tag_stats = {tag: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for tag in self.tag_to_id.keys()}

        with torch.no_grad():
            for sequences, tags, masks in val_loader:
                sequences = sequences.to(self.device)
                tags = tags.to(self.device)
                masks = masks.to(self.device)

                batch_loss = self.model(sequences, tags, masks)
                total_loss += batch_loss.item()
                num_batches += 1

                predictions = self.model.predict(sequences, masks)
                tags_cpu = tags.cpu().numpy()
                masks_cpu = masks.bool().cpu().numpy()

                outside_tag_id = self.tag_to_id[IOBTag.OUTSIDE.value]

                for pred_seq, true_seq, mask_seq in zip(predictions, tags_cpu, masks_cpu):
                    for pred_label, true_label, mask_flag in zip(pred_seq, true_seq, mask_seq):
                        if not mask_flag:
                            continue

                        is_true_positive_label = true_label != outside_tag_id
                        is_pred_positive_label = pred_label != outside_tag_id

                        # Overall metrics
                        if is_true_positive_label and is_pred_positive_label:
                            true_positive += 1
                        elif is_true_positive_label and not is_pred_positive_label:
                            false_negative += 1
                        elif not is_true_positive_label and is_pred_positive_label:
                            false_positive += 1
                        else:
                            true_negative += 1

                        # Per-tag metrics
                        true_tag = self.id_to_tag[true_label]
                        pred_tag = self.id_to_tag[pred_label]

                        if pred_label == true_label:
                            tag_stats[true_tag]["tp"] += 1
                        else:
                            tag_stats[true_tag]["fn"] += 1
                            tag_stats[pred_tag]["fp"] += 1

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        accuracy_denominator = true_positive + false_positive + true_negative + false_negative

        precision = true_positive / precision_denominator if precision_denominator > 0 else 0.0
        recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0
        accuracy = (true_positive + true_negative) / accuracy_denominator if accuracy_denominator > 0 else 0.0

        if precision == 0.0 and recall == 0.0:
            f1_score = 0.0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)

        # Calculate per-tag metrics
        per_tag_metrics = {}
        for tag, stats in tag_stats.items():
            tag_precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
            tag_recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
            tag_f1 = (
                (2 * tag_precision * tag_recall) / (tag_precision + tag_recall) if (tag_precision + tag_recall) > 0 else 0.0
            )

            per_tag_metrics[tag] = {
                "precision": tag_precision,
                "recall": tag_recall,
                "f1": tag_f1,
                "support": stats["tp"] + stats["fn"],
            }

        return {
            "loss": avg_loss,
            "f1": f1_score,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "per_tag_metrics": per_tag_metrics,
        }

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_dir: Optional[str] = None,
    ) -> str:
        """
        Save model checkpoint with versioning and metadata.

        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            checkpoint_dir: Directory to save checkpoint

        Returns:
            Path to saved checkpoint
        """
        if checkpoint_dir is None:
            checkpoint_dir = "checkpoints/field_detector"

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create checkpoint filename with timestamp and metrics
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        f1_score = metrics.get("f1", 0.0)
        checkpoint_name = f"field_detector_v{self.model_version}_epoch{epoch}_f1{f1_score:.4f}_{timestamp}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (self.optimizer.state_dict() if self.optimizer else None),
            "scheduler_state_dict": (self.scheduler.state_dict() if self.scheduler else None),
            "version": self.model_version,
            "timestamp": timestamp,
            "metrics": metrics,
            "config": {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "max_sequence_length": self.max_sequence_length,
            },
            "tag_mappings": {"tag_to_id": self.tag_to_id, "id_to_tag": self.id_to_tag},
        }

        if self.type_classifier:
            checkpoint["type_classifier"] = self.type_classifier

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Also save as "latest" for easy access
        latest_path = os.path.join(checkpoint_dir, f"field_detector_v{self.model_version}_latest.pt")
        torch.save(checkpoint, latest_path)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    async def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = False,
        load_scheduler: bool = False,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint with full state restoration.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to restore optimizer state
            load_scheduler: Whether to restore scheduler state

        Returns:
            Checkpoint metadata
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Restore model state
            if self.model is None:
                # Initialize model with saved config
                config = checkpoint.get("config", {})
                self.model = BiLSTMCRF(
                    vocab_size=config.get("vocab_size", self.vocab_size),
                    embedding_dim=config.get("embedding_dim", self.embedding_dim),
                    hidden_dim=config.get("hidden_dim", self.hidden_dim),
                    num_layers=config.get("num_layers", self.num_layers),
                    num_tags=len(IOBTag),
                ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model_version = checkpoint.get("version", "1.0.0")

            # Restore optimizer state if requested
            if load_optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
                if self.optimizer is None:
                    self.optimizer = torch.optim.AdamW(self.model.parameters())
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Restore scheduler state if requested
            if load_scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
                if self.scheduler is None:
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Restore type classifier if available
            if "type_classifier" in checkpoint:
                self.type_classifier = checkpoint["type_classifier"]

            # Restore tag mappings if available
            if "tag_mappings" in checkpoint:
                self.tag_to_id = checkpoint["tag_mappings"].get("tag_to_id", self.tag_to_id)
                self.id_to_tag = checkpoint["tag_mappings"].get("id_to_tag", self.id_to_tag)

            metadata = {
                "epoch": checkpoint.get("epoch", 0),
                "version": checkpoint.get("version", "1.0.0"),
                "timestamp": checkpoint.get("timestamp", "unknown"),
                "metrics": checkpoint.get("metrics", {}),
                "config": checkpoint.get("config", {}),
            }

            self.logger.info(
                f"Checkpoint loaded from {checkpoint_path}: "
                f"epoch={metadata['epoch']}, f1={metadata['metrics'].get('f1', 0.0):.4f}"
            )

            return metadata

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise ModelException(f"Checkpoint loading failed: {e}")

    def _save_model(self, model_path: Optional[str] = None) -> None:
        """Save model checkpoint."""
        if model_path is None:
            model_path = f"field_detector_v{self.model_version}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "version": self.model_version,
            "config": {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
            },
        }

        if self.type_classifier:
            checkpoint["type_classifier"] = self.type_classifier

        torch.save(checkpoint, model_path)
        self.logger.info(f"Model saved to {model_path}")
