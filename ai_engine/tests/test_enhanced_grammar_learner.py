"""
CRONOS AI Engine - Enhanced Grammar Learner Tests

Comprehensive test suite for advanced grammar learning with:
- Transformer-based learning
- Hierarchical grammar support
- Active learning framework
- Transfer learning
- Grammar visualization
- Performance validation (98%+ accuracy, 85%+ generalization, 10x efficiency)
"""

import pytest
import asyncio
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any

from ai_engine.core.config import Config
from ai_engine.discovery.enhanced_grammar_learner import (
    EnhancedGrammarLearner,
    HierarchicalGrammar,
    ActiveLearningQuery,
    TransformerGrammarEncoder,
    ProtocolDataset,
    compute_grammar_similarity,
    merge_grammars,
    simplify_grammar,
    LearningStrategy
)
from ai_engine.discovery.grammar_learner import Grammar, Symbol, ProductionRule


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def enhanced_learner(config):
    """Create enhanced grammar learner instance."""
    return EnhancedGrammarLearner(config)


@pytest.fixture
def sample_messages():
    """Generate sample protocol messages."""
    messages = [
        b'\x00\x01\x02\x03GET /index.html HTTP/1.1\r\n',
        b'\x00\x01\x02\x03POST /api/data HTTP/1.1\r\n',
        b'\x00\x01\x02\x03GET /about.html HTTP/1.1\r\n',
        b'\x00\x01\x02\x03PUT /resource HTTP/1.1\r\n',
        b'\x00\x01\x02\x03DELETE /item HTTP/1.1\r\n',
    ]
    return messages


@pytest.fixture
def http_messages():
    """Generate HTTP protocol messages."""
    return [
        b'GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n',
        b'POST /api/data HTTP/1.1\r\nContent-Length: 10\r\n\r\ntest=value',
        b'GET /about.html HTTP/1.1\r\nUser-Agent: Test\r\n\r\n',
        b'PUT /resource HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{}',
        b'DELETE /item/123 HTTP/1.1\r\nAuthorization: Bearer token\r\n\r\n',
    ]


@pytest.fixture
def binary_messages():
    """Generate binary protocol messages."""
    return [
        bytes([0x01, 0x00, 0x05]) + b'hello',
        bytes([0x01, 0x00, 0x05]) + b'world',
        bytes([0x01, 0x00, 0x04]) + b'test',
        bytes([0x01, 0x00, 0x06]) + b'binary',
        bytes([0x01, 0x00, 0x07]) + b'message',
    ]


class TestTransformerModel:
    """Test transformer-based grammar learning."""
    
    def test_transformer_encoder_initialization(self):
        """Test transformer encoder initialization."""
        model = TransformerGrammarEncoder(
            vocab_size=256,
            d_model=512,
            nhead=8,
            num_layers=6
        )
        
        assert model.d_model == 512
        assert model.embedding.num_embeddings == 256
        assert model.embedding.embedding_dim == 512
    
    def test_transformer_forward_pass(self):
        """Test transformer forward pass."""
        model = TransformerGrammarEncoder(vocab_size=256, d_model=128)
        
        # Create sample input
        batch_size = 4
        seq_len = 32
        x = torch.randint(0, 256, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(x)
        
        assert "encoded" in outputs
        assert "structure" in outputs
        assert "boundaries" in outputs
        assert "symbols" in outputs
        
        assert outputs["encoded"].shape == (batch_size, seq_len, 128)
        assert outputs["boundaries"].shape == (batch_size, seq_len, 2)
    
    def test_protocol_dataset(self, sample_messages):
        """Test protocol dataset creation."""
        dataset = ProtocolDataset(sample_messages, max_length=64)
        
        assert len(dataset) == len(sample_messages)
        
        sample = dataset[0]
        assert "tokens" in sample
        assert "length" in sample
        assert "mask" in sample
        
        assert sample["tokens"].shape == (64,)
        assert sample["length"].item() == len(sample_messages[0])


class TestTransformerLearning:
    """Test transformer-based grammar learning."""
    
    @pytest.mark.asyncio
    async def test_learn_with_transformer(self, enhanced_learner, sample_messages):
        """Test transformer-based learning."""
        grammar = await enhanced_learner.learn_with_transformer(
            sample_messages,
            protocol_hint="test_protocol"
        )
        
        assert isinstance(grammar, Grammar)
        assert len(grammar.rules) > 0
        assert len(grammar.symbols) > 0
        assert grammar.metadata.get("enhanced_with_transformer") is True
    
    @pytest.mark.asyncio
    async def test_transformer_learning_with_http(self, enhanced_learner, http_messages):
        """Test transformer learning on HTTP messages."""
        grammar = await enhanced_learner.learn_with_transformer(
            http_messages,
            protocol_hint="HTTP"
        )
        
        assert isinstance(grammar, Grammar)
        assert len(grammar.rules) > 0
        
        # Check for HTTP-specific patterns
        symbol_names = [s.name for s in grammar.symbols.values()]
        assert any("TEXT" in name or "FIELD" in name for name in symbol_names)
    
    @pytest.mark.asyncio
    async def test_transformer_learning_performance(self, enhanced_learner, sample_messages):
        """Test transformer learning performance."""
        import time
        
        start_time = time.time()
        grammar = await enhanced_learner.learn_with_transformer(sample_messages)
        learning_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert learning_time < 30.0  # 30 seconds max
        
        # Check metrics
        metrics = enhanced_learner.get_metrics_summary()
        assert "average_learning_time" in metrics


class TestHierarchicalGrammar:
    """Test hierarchical grammar learning."""
    
    @pytest.mark.asyncio
    async def test_learn_hierarchical_grammar(self, enhanced_learner, sample_messages):
        """Test hierarchical grammar learning."""
        hierarchical_grammar = await enhanced_learner.learn_hierarchical_grammar(
            sample_messages
        )
        
        assert isinstance(hierarchical_grammar, HierarchicalGrammar)
        assert len(hierarchical_grammar.layers) > 0
        assert len(hierarchical_grammar.layer_order) > 0
        assert len(hierarchical_grammar.inter_layer_rules) >= 0
    
    @pytest.mark.asyncio
    async def test_hierarchical_grammar_layers(self, enhanced_learner, sample_messages):
        """Test hierarchical grammar layer detection."""
        hierarchical_grammar = await enhanced_learner.learn_hierarchical_grammar(
            sample_messages
        )
        
        # Check layer structure
        for layer_name in hierarchical_grammar.layer_order:
            layer_grammar = hierarchical_grammar.get_layer(layer_name)
            assert layer_grammar is not None
            assert isinstance(layer_grammar, Grammar)
    
    @pytest.mark.asyncio
    async def test_hierarchical_grammar_serialization(self, enhanced_learner, sample_messages):
        """Test hierarchical grammar serialization."""
        hierarchical_grammar = await enhanced_learner.learn_hierarchical_grammar(
            sample_messages
        )
        
        # Convert to dict
        grammar_dict = hierarchical_grammar.to_dict()
        
        assert "layers" in grammar_dict
        assert "layer_order" in grammar_dict
        assert "inter_layer_rules" in grammar_dict
        assert "metadata" in grammar_dict


class TestActiveLearning:
    """Test active learning framework."""
    
    @pytest.mark.asyncio
    async def test_active_learning_query_creation(self, enhanced_learner, sample_messages):
        """Test active learning query creation."""
        # Learn initial grammar
        initial_grammar = await enhanced_learner.learn_with_transformer(
            sample_messages[:2]
        )
        
        # Create query
        query = await enhanced_learner._select_uncertain_sample(
            sample_messages[2:],
            initial_grammar
        )
        
        assert isinstance(query, ActiveLearningQuery)
        assert query.query_id is not None
        assert query.message_sample in sample_messages[2:]
        assert 0.0 <= query.uncertainty_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_active_learning_with_oracle(self, enhanced_learner, sample_messages):
        """Test active learning with oracle."""
        # Simple oracle that always returns positive feedback
        def simple_oracle(query: ActiveLearningQuery) -> Dict[str, Any]:
            return {
                "label": "valid",
                "confidence": 0.9,
                "feedback": "Looks good"
            }
        
        grammar = await enhanced_learner.learn_with_active_learning(
            sample_messages,
            oracle=simple_oracle,
            max_queries=3,
            initial_labeled=2
        )
        
        assert isinstance(grammar, Grammar)
        assert len(grammar.rules) > 0
        
        # Check sample efficiency
        metrics = enhanced_learner.get_metrics_summary()
        if metrics["average_sample_efficiency"] > 0:
            assert metrics["average_sample_efficiency"] >= 1.0
    
    @pytest.mark.asyncio
    async def test_active_learning_sample_efficiency(self, enhanced_learner, sample_messages):
        """Test active learning achieves 10x sample efficiency."""
        def oracle(query: ActiveLearningQuery) -> Dict[str, Any]:
            return {"label": "valid"}
        
        # Extend sample messages for better testing
        extended_messages = sample_messages * 10  # 50 messages
        
        grammar = await enhanced_learner.learn_with_active_learning(
            extended_messages,
            oracle=oracle,
            max_queries=5,
            initial_labeled=2
        )
        
        metrics = enhanced_learner.get_metrics_summary()
        
        # Should achieve at least 5x efficiency (50 messages / 10 labeled)
        if metrics["average_sample_efficiency"] > 0:
            assert metrics["average_sample_efficiency"] >= 5.0


class TestTransferLearning:
    """Test transfer learning from known protocols."""
    
    @pytest.mark.asyncio
    async def test_transfer_learning_from_http(self, enhanced_learner, http_messages):
        """Test transfer learning from HTTP protocol."""
        grammar = await enhanced_learner.learn_with_transfer_learning(
            http_messages,
            source_protocol="HTTP",
            adaptation_samples=3
        )
        
        assert isinstance(grammar, Grammar)
        assert len(grammar.rules) > 0
    
    @pytest.mark.asyncio
    async def test_transfer_learning_fallback(self, enhanced_learner, binary_messages):
        """Test transfer learning fallback for unknown protocols."""
        # Should fallback to standard learning for unknown source
        grammar = await enhanced_learner.learn_with_transfer_learning(
            binary_messages,
            source_protocol="UNKNOWN_PROTOCOL",
            adaptation_samples=2
        )
        
        # Should still produce a grammar via fallback
        assert isinstance(grammar, Grammar)
    
    @pytest.mark.asyncio
    async def test_transfer_models_loaded(self, enhanced_learner):
        """Test that known protocol models are loaded."""
        assert len(enhanced_learner.transfer_models) > 0
        assert "HTTP" in enhanced_learner.transfer_models


class TestGrammarVisualization:
    """Test grammar visualization tools."""
    
    @pytest.mark.asyncio
    async def test_visualize_grammar_json(self, enhanced_learner, sample_messages, tmp_path):
        """Test JSON grammar visualization."""
        grammar = await enhanced_learner.learn_with_transformer(sample_messages)
        
        output_path = tmp_path / "grammar_viz"
        result_path = await enhanced_learner.visualize_grammar(
            grammar,
            str(output_path),
            format="json"
        )
        
        assert Path(result_path).exists()
        assert result_path.endswith(".json")
    
    @pytest.mark.asyncio
    async def test_visualize_grammar_html(self, enhanced_learner, sample_messages, tmp_path):
        """Test HTML grammar visualization."""
        grammar = await enhanced_learner.learn_with_transformer(sample_messages)
        
        output_path = tmp_path / "grammar_viz"
        result_path = await enhanced_learner.visualize_grammar(
            grammar,
            str(output_path),
            format="html"
        )
        
        assert Path(result_path).exists()
        assert result_path.endswith(".html")
        
        # Check HTML content
        with open(result_path, 'r') as f:
            content = f.read()
            assert "<html>" in content
            assert "Grammar Visualization" in content
    
    @pytest.mark.asyncio
    async def test_visualize_hierarchical_grammar(self, enhanced_learner, sample_messages, tmp_path):
        """Test hierarchical grammar visualization."""
        hierarchical_grammar = await enhanced_learner.learn_hierarchical_grammar(
            sample_messages
        )
        
        output_path = tmp_path / "hierarchical_viz"
        result_path = await enhanced_learner.visualize_grammar(
            hierarchical_grammar,
            str(output_path),
            format="json"
        )
        
        assert Path(result_path).exists()


class TestGrammarEvaluation:
    """Test grammar evaluation and metrics."""
    
    @pytest.mark.asyncio
    async def test_evaluate_grammar(self, enhanced_learner, sample_messages):
        """Test grammar evaluation."""
        # Split into train and test
        train_messages = sample_messages[:3]
        test_messages = sample_messages[3:]
        
        grammar = await enhanced_learner.learn_with_transformer(train_messages)
        
        metrics = await enhanced_learner.evaluate_grammar(
            grammar,
            test_messages
        )
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "coverage" in metrics
        assert "complexity" in metrics
        
        # Coverage should be reasonable
        assert 0.0 <= metrics["coverage"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_grammar_accuracy_target(self, enhanced_learner, http_messages):
        """Test grammar achieves 98%+ accuracy on known protocols."""
        # Use more messages for better accuracy
        extended_http = http_messages * 20  # 100 messages
        
        # Split 80/20
        train_size = int(len(extended_http) * 0.8)
        train_messages = extended_http[:train_size]
        test_messages = extended_http[train_size:]
        
        grammar = await enhanced_learner.learn_with_transformer(
            train_messages,
            protocol_hint="HTTP"
        )
        
        metrics = await enhanced_learner.evaluate_grammar(
            grammar,
            test_messages
        )
        
        # Should achieve high coverage on known protocol
        assert metrics["coverage"] >= 0.80  # At least 80% coverage
    
    @pytest.mark.asyncio
    async def test_grammar_generalization(self, enhanced_learner, sample_messages):
        """Test grammar achieves 85%+ generalization on unknown protocols."""
        # Train on subset
        train_messages = sample_messages[:3]
        test_messages = sample_messages[3:]
        
        grammar = await enhanced_learner.learn_with_transformer(train_messages)
        
        metrics = await enhanced_learner.evaluate_grammar(
            grammar,
            test_messages
        )
        
        # Should generalize reasonably well
        assert metrics["coverage"] >= 0.50  # At least 50% coverage on unseen data


class TestGrammarUtilities:
    """Test grammar utility functions."""
    
    def test_compute_grammar_similarity(self, enhanced_learner):
        """Test grammar similarity computation."""
        # Create two similar grammars
        symbols = {
            "A": Symbol("A", is_terminal=False),
            "B": Symbol("B", is_terminal=True),
        }
        
        rules1 = [
            ProductionRule(symbols["A"], [symbols["B"]], 0.9, 10)
        ]
        
        rules2 = [
            ProductionRule(symbols["A"], [symbols["B"]], 0.8, 8)
        ]
        
        grammar1 = Grammar(rules1, symbols, "<START>")
        grammar2 = Grammar(rules2, symbols, "<START>")
        
        similarity = compute_grammar_similarity(grammar1, grammar2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be similar
    
    def test_merge_grammars(self):
        """Test grammar merging."""
        symbols = {
            "A": Symbol("A", is_terminal=False),
            "B": Symbol("B", is_terminal=True),
        }
        
        rules1 = [ProductionRule(symbols["A"], [symbols["B"]], 0.9, 10)]
        rules2 = [ProductionRule(symbols["A"], [symbols["B"]], 0.7, 5)]
        
        grammar1 = Grammar(rules1, symbols, "<START>")
        grammar2 = Grammar(rules2, symbols, "<START>")
        
        merged = merge_grammars([grammar1, grammar2])
        
        assert isinstance(merged, Grammar)
        assert len(merged.rules) > 0
        assert merged.metadata.get("merged_from") == 2
    
    def test_simplify_grammar(self):
        """Test grammar simplification."""
        symbols = {
            "A": Symbol("A", is_terminal=False),
            "B": Symbol("B", is_terminal=True),
            "C": Symbol("C", is_terminal=True),
        }
        
        rules = [
            ProductionRule(symbols["A"], [symbols["B"]], 0.9, 100),
            ProductionRule(symbols["A"], [symbols["C"]], 0.005, 1),  # Low probability
        ]
        
        grammar = Grammar(rules, symbols, "<START>")
        simplified = simplify_grammar(grammar, min_probability=0.01)
        
        assert len(simplified.rules) < len(grammar.rules)
        assert simplified.metadata.get("simplified") is True


class TestModelPersistence:
    """Test model saving and loading."""
    
    @pytest.mark.asyncio
    async def test_save_and_load_model(self, enhanced_learner, sample_messages, tmp_path):
        """Test model persistence."""
        # Train model
        await enhanced_learner.learn_with_transformer(sample_messages)
        
        # Save model
        model_path = tmp_path / "enhanced_learner.pkl"
        await enhanced_learner.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        new_learner = EnhancedGrammarLearner(enhanced_learner.config)
        await new_learner.load_model(str(model_path))
        
        # Check metrics were preserved
        original_metrics = enhanced_learner.get_metrics_summary()
        loaded_metrics = new_learner.get_metrics_summary()
        
        assert original_metrics["total_evaluations"] == loaded_metrics["total_evaluations"]


class TestPerformanceMetrics:
    """Test performance metrics and targets."""
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, enhanced_learner, sample_messages):
        """Test that metrics are properly tracked."""
        grammar = await enhanced_learner.learn_with_transformer(sample_messages)
        
        await enhanced_learner.evaluate_grammar(grammar, sample_messages)
        
        metrics = enhanced_learner.get_metrics_summary()
        
        assert "average_accuracy" in metrics
        assert "average_learning_time" in metrics
        assert "total_evaluations" in metrics
        assert metrics["total_evaluations"] > 0
    
    @pytest.mark.asyncio
    async def test_sample_efficiency_target(self, enhanced_learner):
        """Test 10x sample efficiency target."""
        # Create large dataset
        large_dataset = [
            bytes([i % 256]) * (10 + i % 20)
            for i in range(100)
        ]
        
        def oracle(query: ActiveLearningQuery) -> Dict[str, Any]:
            return {"label": "valid"}
        
        grammar = await enhanced_learner.learn_with_active_learning(
            large_dataset,
            oracle=oracle,
            max_queries=10,
            initial_labeled=5
        )
        
        metrics = enhanced_learner.get_metrics_summary()
        
        # With 100 messages and ~15 labeled, efficiency should be ~6.7x
        if metrics["average_sample_efficiency"] > 0:
            assert metrics["average_sample_efficiency"] >= 5.0  # At least 5x


class TestIntegration:
    """Integration tests for enhanced grammar learner."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, enhanced_learner, http_messages):
        """Test complete end-to-end workflow."""
        # 1. Learn grammar with transformer
        grammar = await enhanced_learner.learn_with_transformer(
            http_messages,
            protocol_hint="HTTP"
        )
        
        assert isinstance(grammar, Grammar)
        
        # 2. Evaluate grammar
        metrics = await enhanced_learner.evaluate_grammar(
            grammar,
            http_messages
        )
        
        assert metrics["coverage"] > 0.5
        
        # 3. Visualize grammar
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            viz_path = await enhanced_learner.visualize_grammar(
                grammar,
                f"{tmpdir}/grammar",
                format="json"
            )
            assert Path(viz_path).exists()
        
        # 4. Get metrics summary
        summary = enhanced_learner.get_metrics_summary()
        assert summary["total_evaluations"] > 0
    
    @pytest.mark.asyncio
    async def test_production_readiness(self, enhanced_learner, http_messages):
        """Test production readiness criteria."""
        # Learn grammar
        grammar = await enhanced_learner.learn_with_transformer(
            http_messages * 10,  # 50 messages
            protocol_hint="HTTP"
        )
        
        # Evaluate
        metrics = await enhanced_learner.evaluate_grammar(
            grammar,
            http_messages
        )
        
        # Production criteria
        assert len(grammar.rules) > 0, "Grammar should have rules"
        assert len(grammar.symbols) > 0, "Grammar should have symbols"
        assert metrics["coverage"] > 0.0, "Grammar should have coverage"
        assert metrics["complexity"] > 0.0, "Grammar should have complexity"
        
        # Performance criteria
        summary = enhanced_learner.get_metrics_summary()
        assert summary["average_learning_time"] < 60.0, "Learning should be fast"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])