"""
CRONOS AI Engine - Field Detector

This module implements BiLSTM-CRF based field detection for automatic
identification of field boundaries and types in protocol messages.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from torchcrf import CRF
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
        char_vocab_size: int = 256
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
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to map LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)
        
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
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        
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
        return -self.crf(lstm_feats, tags, mask=masks.bool())
    
    def predict(self, sequences: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        """Predict tag sequences."""
        lstm_feats = self._get_lstm_features(sequences, masks)
        return self.crf.decode(lstm_feats, mask=masks.bool())


class FieldDetector:
    """
    Main field detection system using BiLSTM-CRF.
    
    This class provides high-level interface for field detection,
    including training, inference, and post-processing.
    """
    
    def __init__(self, config: Config):
        """Initialize field detector."""
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
        self.id_to_tag = {idx: tag.value for tag, idx in self.tag_to_id.items()}
        
        # Field type classifier (separate model for type inference)
        self.type_classifier = None
        
        # Model components
        self.model: Optional[BiLSTMCRF] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                num_tags=len(IOBTag)
            ).to(self.device)
            
            # Initialize type classifier
            await self._initialize_type_classifier()
            
            self.logger.info("FieldDetector model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FieldDetector: {e}")
            raise ModelException(f"FieldDetector initialization failed: {e}")
    
    async def detect_boundaries(
        self,
        message_data: bytes,
        protocol_type: Optional[str] = None
    ) -> List[FieldBoundary]:
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
            field_boundaries = self._postprocess_predictions(
                predictions[0], message_data, protocol_type
            )
            
            return field_boundaries
            
        except Exception as e:
            self.logger.error(f"Field boundary detection failed: {e}")
            raise FieldDetectionException(f"Boundary detection error: {e}")
    
    async def infer_field_types(self, field_boundaries: List[FieldBoundary]) -> List[FieldBoundary]:
        """Infer field types for detected boundaries."""
        if not self.type_classifier:
            # Use simple heuristic-based type inference
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
        batch_size: int = 16
    ) -> Dict[str, List[float]]:
        """
        Train the field detection model.
        
        Args:
            training_data: List of (message, field_annotations) pairs
            validation_data: Optional validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            
        Returns:
            Training history with losses and metrics
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=True
        )
        
        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_val_f1 = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_loader:
                val_loss, val_f1 = self._validate_epoch(val_loader)
                history["val_loss"].append(val_loss)
                history["val_f1"].append(val_f1)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    await self._save_model()
                
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}"
                )
            else:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}")
        
        self.logger.info(f"Training completed. Best validation F1: {best_val_f1:.4f}")
        return history
    
    async def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_version = checkpoint.get('version', '1.0.0')
            
            if 'type_classifier' in checkpoint:
                self.type_classifier = checkpoint['type_classifier']
            
            self.logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelException(f"Model loading failed: {e}")
    
    async def save_model(self, model_path: str) -> None:
        """Save the trained model."""
        await self._save_model(model_path)
    
    def _preprocess_message(self, message_data: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess message data for model input."""
        # Convert bytes to sequence of integers
        byte_sequence = list(message_data)
        
        # Truncate or pad to max_sequence_length
        if len(byte_sequence) > self.max_sequence_length:
            byte_sequence = byte_sequence[:self.max_sequence_length]
        
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
        self,
        predictions: List[int],
        message_data: bytes,
        protocol_type: Optional[str]
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
                    field_boundaries.append(FieldBoundary(
                        start_pos=current_field_start,
                        end_pos=i,
                        field_type=FieldType.UNKNOWN,
                        confidence=0.8,  # Default confidence
                        raw_value=message_data[current_field_start:i]
                    ))
                current_field_start = i
                
            elif tag == IOBTag.OUTSIDE.value:
                # End of field
                if current_field_start is not None:
                    field_boundaries.append(FieldBoundary(
                        start_pos=current_field_start,
                        end_pos=i,
                        field_type=FieldType.UNKNOWN,
                        confidence=0.8,
                        raw_value=message_data[current_field_start:i]
                    ))
                    current_field_start = None
            
            # IOBTag.INSIDE_FIELD continues current field
        
        # Handle field that extends to end of message
        if current_field_start is not None:
            field_boundaries.append(FieldBoundary(
                start_pos=current_field_start,
                end_pos=len(message_data),
                field_type=FieldType.UNKNOWN,
                confidence=0.8,
                raw_value=message_data[current_field_start:]
            ))
        
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
            elif field_data == b'\x00' * field_length:
                boundary.field_type = FieldType.RESERVED
            else:
                boundary.field_type = FieldType.BINARY
        
        return field_boundaries
    
    async def _predict_field_type(self, field_data: bytes) -> FieldType:
        """Predict field type using ML classifier."""
        # TODO: Implement ML-based type classifier
        # For now, fall back to heuristics
        
        field_length = len(field_data)
        
        if field_length == 4 and all(b < 128 for b in field_data):
            return FieldType.INTEGER
        elif all(32 <= b <= 126 for b in field_data):
            return FieldType.STRING
        else:
            return FieldType.BINARY
    
    async def _initialize_type_classifier(self) -> None:
        """Initialize field type classifier."""
        # TODO: Implement separate field type classifier
        self.type_classifier = None
        self.logger.info("Type classifier initialized (heuristic-based)")
    
    def _create_data_loader(
        self,
        data: List[Tuple[bytes, List[Tuple[int, int, str]]]],
        batch_size: int,
        shuffle: bool = True
    ):
        """Create data loader for training/validation."""
        # TODO: Implement proper DataLoader
        # This is a simplified version
        return data
    
    def _train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Simplified training loop
        for batch_data in train_loader:
            self.optimizer.zero_grad()
            
            # TODO: Implement proper batch processing
            # This is simplified for demonstration
            
            # Dummy loss calculation
            loss = torch.tensor(0.5, requires_grad=True)
            loss.backward()
            
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, val_loader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        f1_score = 0.8  # Placeholder
        
        with torch.no_grad():
            # TODO: Implement proper validation
            pass
        
        return total_loss, f1_score
    
    async def _save_model(self, model_path: Optional[str] = None) -> None:
        """Save model checkpoint."""
        if model_path is None:
            model_path = f"field_detector_v{self.model_version}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'version': self.model_version,
            'config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
            }
        }
        
        if self.type_classifier:
            checkpoint['type_classifier'] = self.type_classifier
        
        torch.save(checkpoint, model_path)
        self.logger.info(f"Model saved to {model_path}")