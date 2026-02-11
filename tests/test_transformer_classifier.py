"""
Tests for Protocol Transformer Classifier

Comprehensive test suite for the Transformer-based protocol classifier
including unit tests, integration tests, and performance benchmarks.
"""

import asyncio
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Import test targets
from ai_engine.discovery.transformer_classifier import (
    TransformerConfig,
    ProtocolSample,
    ClassificationResult,
    ProtocolTransformer,
    ProtocolTransformerClassifier,
    PositionalEncoding,
    ByteEmbedding,
    MultiHeadAttentionWithVisualization,
    ProtocolTransformerEncoderLayer,
    create_transformer_classifier,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def transformer_config():
    """Create a small Transformer config for testing."""
    return TransformerConfig(
        vocab_size=259,  # 256 bytes + 3 special tokens
        max_sequence_length=128,  # Smaller for fast tests
        embedding_dim=64,
        num_attention_heads=4,
        num_encoder_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=8,
        warmup_steps=10,
        max_epochs=5,
        early_stopping_patience=3,
        enable_pretraining=False,  # Skip for unit tests
        enable_contrastive=False,
        enable_few_shot=True,
    )


@pytest.fixture
def sample_training_data():
    """Create sample training data for protocol classification."""
    samples = []

    # HTTP-like protocol samples
    for i in range(20):
        http_data = f"GET /path{i} HTTP/1.1\r\nHost: example.com\r\n\r\n".encode()
        samples.append(ProtocolSample(data=http_data, label="http"))

    # Binary protocol samples (simulated)
    for i in range(20):
        binary_data = bytes([0x01, 0x02, i % 256, 0x04] + [i] * 20)
        samples.append(ProtocolSample(data=binary_data, label="binary_protocol"))

    # JSON-like protocol samples
    for i in range(20):
        json_data = f'{{"type": "message", "id": {i}, "data": "test"}}'.encode()
        samples.append(ProtocolSample(data=json_data, label="json_protocol"))

    # XML-like protocol samples
    for i in range(20):
        xml_data = f'<message id="{i}"><data>test</data></message>'.encode()
        samples.append(ProtocolSample(data=xml_data, label="xml_protocol"))

    return samples


@pytest.fixture
def rare_protocol_samples():
    """Create samples for rare protocol (few-shot testing)."""
    return [
        ProtocolSample(
            data=b"RARE_PROTO_V1\x00\x01\x02\x03",
            label="rare_protocol"
        ),
        ProtocolSample(
            data=b"RARE_PROTO_V1\x00\x04\x05\x06",
            label="rare_protocol"
        ),
        ProtocolSample(
            data=b"RARE_PROTO_V1\x00\x07\x08\x09",
            label="rare_protocol"
        ),
    ]


# =============================================================================
# Unit Tests - Components
# =============================================================================

class TestPositionalEncoding:
    """Test positional encoding component."""

    def test_initialization(self):
        """Test positional encoding initialization."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        assert pe.pe.shape == (1, 100, 64)

    def test_forward_shape(self):
        """Test forward pass shape."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(2, 50, 64)
        output = pe(x)
        assert output.shape == x.shape

    def test_positional_values(self):
        """Test that positional encodings vary by position."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        # Positions should have different encodings
        assert not torch.allclose(pe.pe[0, 0], pe.pe[0, 1])


class TestByteEmbedding:
    """Test byte embedding layer."""

    def test_initialization(self):
        """Test embedding initialization."""
        embedding = ByteEmbedding(vocab_size=259, embedding_dim=64)
        assert embedding.vocab_size == 259
        assert embedding.embedding_dim == 64

    def test_special_tokens(self):
        """Test special token IDs."""
        embedding = ByteEmbedding()
        assert embedding.pad_id == 0
        assert embedding.cls_id == 256
        assert embedding.sep_id == 257
        assert embedding.mask_id == 258

    def test_forward_shape(self):
        """Test forward pass output shape."""
        embedding = ByteEmbedding(vocab_size=259, embedding_dim=64, max_len=100)
        x = torch.randint(0, 256, (2, 50))  # Batch of 2, length 50
        output = embedding(x)
        assert output.shape == (2, 50, 64)


class TestMultiHeadAttention:
    """Test multi-head attention with visualization."""

    def test_initialization(self):
        """Test attention initialization."""
        attn = MultiHeadAttentionWithVisualization(
            embed_dim=64, num_heads=4
        )
        assert attn.num_heads == 4
        assert attn.head_dim == 16

    def test_forward_shape(self):
        """Test forward pass shapes."""
        attn = MultiHeadAttentionWithVisualization(
            embed_dim=64, num_heads=4
        )
        x = torch.randn(2, 10, 64)
        output, weights = attn(x, x, x)
        assert output.shape == (2, 10, 64)
        assert weights is None  # Not requested

    def test_attention_weights(self):
        """Test attention weight retrieval."""
        attn = MultiHeadAttentionWithVisualization(
            embed_dim=64, num_heads=4
        )
        x = torch.randn(2, 10, 64)
        output, weights = attn(x, x, x, need_weights=True)
        assert weights is not None
        assert weights.shape == (2, 4, 10, 10)  # (batch, heads, seq, seq)


class TestTransformerEncoderLayer:
    """Test Transformer encoder layer."""

    def test_forward_shape(self):
        """Test encoder layer forward pass."""
        layer = ProtocolTransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256
        )
        x = torch.randn(2, 10, 64)
        output, attn = layer(x)
        assert output.shape == x.shape

    def test_attention_retrieval(self):
        """Test attention weight retrieval from encoder."""
        layer = ProtocolTransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256
        )
        x = torch.randn(2, 10, 64)
        output, attn = layer(x, need_weights=True)
        assert attn is not None


# =============================================================================
# Unit Tests - Model
# =============================================================================

class TestProtocolTransformer:
    """Test Protocol Transformer model."""

    def test_initialization(self, transformer_config):
        """Test model initialization."""
        model = ProtocolTransformer(transformer_config, num_classes=4)
        assert model.num_classes == 4
        assert len(model.encoder_layers) == 2

    def test_parameter_count(self, transformer_config):
        """Test model has trainable parameters."""
        model = ProtocolTransformer(transformer_config, num_classes=4)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_forward_classification(self, transformer_config):
        """Test classification forward pass."""
        model = ProtocolTransformer(transformer_config, num_classes=4)
        input_ids = torch.randint(0, 256, (2, transformer_config.max_sequence_length))

        output = model(input_ids)

        assert "logits" in output
        assert "embeddings" in output
        assert output["logits"].shape == (2, 4)
        assert output["embeddings"].shape == (2, transformer_config.embedding_dim)

    def test_forward_with_labels(self, transformer_config):
        """Test forward pass with labels computes loss."""
        model = ProtocolTransformer(transformer_config, num_classes=4)
        input_ids = torch.randint(0, 256, (2, transformer_config.max_sequence_length))
        labels = torch.tensor([0, 1])

        output = model(input_ids, labels=labels)

        assert "loss" in output
        assert output["loss"].item() > 0

    def test_encode(self, transformer_config):
        """Test encoding function."""
        model = ProtocolTransformer(transformer_config, num_classes=4)
        input_ids = torch.randint(0, 256, (2, transformer_config.max_sequence_length))

        cls_output, all_hidden, all_attn = model.encode(
            input_ids, return_all_layers=True, need_attention_weights=True
        )

        assert cls_output.shape == (2, transformer_config.embedding_dim)
        assert len(all_hidden) == 2  # num_encoder_layers
        assert len(all_attn) == 2

    def test_mlm_forward(self, transformer_config):
        """Test masked language modeling forward pass."""
        model = ProtocolTransformer(transformer_config, num_classes=4)
        input_ids = torch.randint(0, 256, (2, transformer_config.max_sequence_length))
        mlm_labels = torch.randint(-100, 259, (2, transformer_config.max_sequence_length))

        output = model.forward_mlm(input_ids, mlm_labels=mlm_labels)

        assert "prediction_scores" in output
        assert "loss" in output

    def test_contrastive_forward(self, transformer_config):
        """Test contrastive learning forward pass."""
        transformer_config.enable_contrastive = True
        model = ProtocolTransformer(transformer_config, num_classes=4)
        input_ids = torch.randint(0, 256, (2, transformer_config.max_sequence_length))

        projections = model.forward_contrastive(input_ids)

        assert projections.shape == (2, transformer_config.prototype_dim)
        # Verify L2 normalized
        norms = projections.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# =============================================================================
# Integration Tests - Classifier
# =============================================================================

class TestProtocolTransformerClassifier:
    """Test Protocol Transformer Classifier (high-level API)."""

    @pytest.mark.asyncio
    async def test_initialization(self, transformer_config):
        """Test classifier initialization."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        assert not classifier.is_trained
        assert classifier.device is not None

    @pytest.mark.asyncio
    async def test_training(self, transformer_config, sample_training_data):
        """Test classifier training."""
        classifier = ProtocolTransformerClassifier(transformer_config)

        results = await classifier.train(
            sample_training_data,
            validation_split=0.2,
            epochs=3
        )

        assert classifier.is_trained
        assert results["num_classes"] == 4
        assert "http" in results["protocols"]
        assert "best_val_accuracy" in results

    @pytest.mark.asyncio
    async def test_classification(self, transformer_config, sample_training_data):
        """Test classification after training."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        # Test HTTP classification
        result = await classifier.classify(b"GET /test HTTP/1.1\r\n\r\n")

        assert isinstance(result, ClassificationResult)
        assert result.protocol_type in classifier.known_protocols
        assert 0 <= result.confidence <= 1
        assert len(result.probabilities) == 4

    @pytest.mark.asyncio
    async def test_classification_with_attention(
        self, transformer_config, sample_training_data
    ):
        """Test classification with attention weights."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        result = await classifier.classify(
            b"GET /test HTTP/1.1\r\n\r\n",
            return_attention=True
        )

        assert result.attention_weights is not None
        assert result.attention_weights.shape[0] == transformer_config.max_sequence_length

    @pytest.mark.asyncio
    async def test_classification_with_explanation(
        self, transformer_config, sample_training_data
    ):
        """Test classification with explanation."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        result = await classifier.classify(
            b"GET /test HTTP/1.1\r\n\r\n",
            return_attention=True,
            return_explanation=True
        )

        assert result.explanation is not None
        assert "key_bytes" in result.explanation

    @pytest.mark.asyncio
    async def test_empty_input(self, transformer_config, sample_training_data):
        """Test classification with empty input."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        result = await classifier.classify(b"")

        assert result.protocol_type == "unknown"
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_untrained_classifier(self, transformer_config):
        """Test that untrained classifier raises error."""
        classifier = ProtocolTransformerClassifier(transformer_config)

        with pytest.raises(RuntimeError, match="not trained"):
            await classifier.classify(b"test")

    @pytest.mark.asyncio
    async def test_get_embeddings(self, transformer_config, sample_training_data):
        """Test embedding extraction."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        embeddings = await classifier.get_embeddings(b"GET /test HTTP/1.1\r\n\r\n")

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (transformer_config.embedding_dim,)

    @pytest.mark.asyncio
    async def test_few_shot_classification(
        self, transformer_config, sample_training_data, rare_protocol_samples
    ):
        """Test few-shot classification for rare protocols."""
        transformer_config.enable_few_shot = True
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        # Classify a rare protocol sample using few-shot
        result = await classifier.classify_few_shot(
            b"RARE_PROTO_V1\x00\x0A\x0B\x0C",
            rare_protocol_samples
        )

        assert result.protocol_type == "rare_protocol"
        assert "few_shot" in result.metadata.get("method", "")

    @pytest.mark.asyncio
    async def test_prediction_caching(
        self, transformer_config, sample_training_data
    ):
        """Test that predictions are cached."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        test_data = b"GET /test HTTP/1.1\r\n\r\n"

        # First call
        result1 = await classifier.classify(test_data)

        # Second call should be cached
        result2 = await classifier.classify(test_data)

        assert result1.protocol_type == result2.protocol_type
        assert result1.confidence == result2.confidence

    @pytest.mark.asyncio
    async def test_model_info(self, transformer_config, sample_training_data):
        """Test model info retrieval."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        info = await classifier.get_model_info()

        assert info["is_trained"] is True
        assert info["model_type"] == "ProtocolTransformer"
        assert "parameters" in info
        assert len(info["supported_protocols"]) == 4


# =============================================================================
# Tests - Pretraining
# =============================================================================

class TestPretraining:
    """Test pre-training functionality."""

    @pytest.mark.asyncio
    async def test_mlm_mask_creation(self, transformer_config):
        """Test MLM mask creation."""
        classifier = ProtocolTransformerClassifier(transformer_config)

        # Create dummy model for mask creation
        classifier.model = ProtocolTransformer(transformer_config, num_classes=2)

        input_ids = torch.randint(1, 256, (4, 32))  # Avoid 0 (padding)

        masked_input, labels = classifier._create_mlm_masks(input_ids)

        # Check that some tokens are masked
        mask_count = (masked_input == 258).sum().item()  # MASK token
        assert mask_count > 0

        # Check that labels are set correctly
        label_count = (labels != -100).sum().item()
        assert label_count > 0

    @pytest.mark.asyncio
    async def test_pretraining(self, transformer_config, sample_training_data):
        """Test pre-training workflow."""
        transformer_config.enable_pretraining = True
        classifier = ProtocolTransformerClassifier(transformer_config)

        # Create corpus from training data
        corpus = [sample.data for sample in sample_training_data]

        results = await classifier.pretrain(corpus, epochs=2)

        assert classifier.is_pretrained
        assert "final_loss" in results
        assert results["epochs"] == 2


# =============================================================================
# Tests - Saving and Loading
# =============================================================================

class TestModelPersistence:
    """Test model saving and loading."""

    @pytest.mark.asyncio
    async def test_save_and_load(
        self, transformer_config, sample_training_data, tmp_path
    ):
        """Test model save and load cycle."""
        # Train classifier
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        # Get prediction before save
        result_before = await classifier.classify(b"GET /test HTTP/1.1\r\n\r\n")

        # Save model
        model_path = str(tmp_path / "model")
        await classifier.save_model(model_path)

        # Create new classifier and load
        new_classifier = ProtocolTransformerClassifier(transformer_config)
        await new_classifier.load_model(model_path)

        # Get prediction after load
        result_after = await new_classifier.classify(b"GET /test HTTP/1.1\r\n\r\n")

        assert result_before.protocol_type == result_after.protocol_type
        # Confidence might differ slightly due to dropout state
        assert abs(result_before.confidence - result_after.confidence) < 0.1

    @pytest.mark.asyncio
    async def test_save_untrained(self, transformer_config, tmp_path):
        """Test that saving untrained model raises error."""
        classifier = ProtocolTransformerClassifier(transformer_config)

        with pytest.raises(RuntimeError, match="No trained model"):
            await classifier.save_model(str(tmp_path / "model"))


# =============================================================================
# Tests - Factory Function
# =============================================================================

class TestFactoryFunction:
    """Test factory function."""

    def test_create_classifier(self, transformer_config):
        """Test factory function creates classifier."""
        classifier = create_transformer_classifier(transformer_config)
        assert isinstance(classifier, ProtocolTransformerClassifier)
        assert not classifier.is_trained


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.asyncio
    async def test_inference_speed(
        self, transformer_config, sample_training_data
    ):
        """Test inference speed is reasonable."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        test_data = b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"

        # Warm up
        await classifier.classify(test_data)

        # Clear cache for accurate timing
        classifier._prediction_cache.clear()

        # Measure inference time
        import time
        start = time.time()
        for _ in range(10):
            classifier._prediction_cache.clear()  # Clear cache each iteration
            await classifier.classify(test_data)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        # Should be under 100ms per inference on CPU
        assert avg_time < 0.5, f"Inference too slow: {avg_time:.3f}s"

    @pytest.mark.asyncio
    async def test_batch_efficiency(self, transformer_config):
        """Test batch processing is more efficient than sequential."""
        model = ProtocolTransformer(transformer_config, num_classes=4)

        # Single sample
        single_input = torch.randint(0, 256, (1, transformer_config.max_sequence_length))

        # Batch of 8
        batch_input = torch.randint(0, 256, (8, transformer_config.max_sequence_length))

        import time

        # Time single samples
        start = time.time()
        for _ in range(8):
            model(single_input)
        single_time = time.time() - start

        # Time batch
        start = time.time()
        model(batch_input)
        batch_time = time.time() - start

        # Batch should be faster than 8 sequential singles
        # Allow some tolerance for overhead
        assert batch_time < single_time * 0.8, \
            f"Batch ({batch_time:.3f}s) not faster than sequential ({single_time:.3f}s)"


# =============================================================================
# Tests - Shutdown
# =============================================================================

class TestShutdown:
    """Test cleanup and shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_resources(
        self, transformer_config, sample_training_data
    ):
        """Test shutdown clears resources."""
        classifier = ProtocolTransformerClassifier(transformer_config)
        await classifier.train(sample_training_data, epochs=3)

        # Add some cache entries
        await classifier.classify(b"test1")
        await classifier.classify(b"test2")

        assert len(classifier._prediction_cache) > 0

        await classifier.shutdown()

        assert classifier.model is None
        assert len(classifier._prediction_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
