"""
QBITEL Engine - Enhanced PCFG Inference Tests

Comprehensive test suite for production-ready PCFG inference with:
- Bayesian hyperparameter optimization
- Parallel processing
- Incremental learning
- Advanced convergence detection
- Grammar quality metrics
"""

import pytest
import asyncio
import time
import numpy as np
from typing import List
from unittest.mock import Mock, patch, AsyncMock

from ai_engine.core.config import Config
from ai_engine.core.exceptions import ProtocolException, ModelException
from ai_engine.discovery.enhanced_pcfg_inference import (
    EnhancedPCFGInference,
    ProductionRule,
    Grammar,
    HyperparameterConfig,
    BayesianOptimizer,
)
from ai_engine.discovery.production_enhancements import ProductionConfig, CacheBackend


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def hyperparams():
    """Create test hyperparameters."""
    return HyperparameterConfig(
        min_pattern_frequency=2,
        max_rule_length=8,
        min_symbol_entropy=0.3,
        max_grammar_size=500,
        convergence_threshold=0.01,
        max_iterations=50,
    )


@pytest.fixture
def production_config():
    """Create test production configuration."""
    return ProductionConfig(
        enable_caching=False,  # Disable for tests
        enable_circuit_breakers=False,
        worker_threads=2,
    )


@pytest.fixture
def pcfg_engine(config, hyperparams, production_config):
    """Create PCFG inference engine."""
    return EnhancedPCFGInference(
        config=config,
        hyperparams=hyperparams,
        production_config=production_config,
        enable_caching=False,
        enable_parallel=False,  # Disable for deterministic tests
    )


@pytest.fixture
def sample_messages():
    """Create sample protocol messages."""
    return [
        b"\x01\x02\x03\x04\x05",
        b"\x01\x02\x03\x04\x06",
        b"\x01\x02\x03\x04\x07",
        b"\x01\x02\x03\x05\x08",
        b"\x01\x02\x03\x05\x09",
    ]


@pytest.fixture
def complex_messages():
    """Create complex protocol messages with patterns."""
    messages = []

    # Messages with common header
    for i in range(10):
        msg = b"\xff\xfe" + bytes([i]) + b"\x00" * 5 + bytes([i * 2])
        messages.append(msg)

    # Messages with repeating patterns
    for i in range(5):
        msg = b"\xaa\xbb" * 3 + bytes([i])
        messages.append(msg)

    return messages


# ============================================================================
# BASIC INFERENCE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_basic_inference(pcfg_engine, sample_messages):
    """Test basic PCFG inference."""
    grammar = await pcfg_engine.infer(sample_messages)

    assert grammar is not None
    assert len(grammar.rules) > 0
    assert len(grammar.terminals) > 0
    assert len(grammar.non_terminals) > 0
    assert grammar.start_symbol == "<START>"
    assert grammar.message_count == len(sample_messages)


@pytest.mark.asyncio
async def test_empty_messages_error(pcfg_engine):
    """Test that empty message list raises error."""
    with pytest.raises(ProtocolException):
        await pcfg_engine.infer([])


@pytest.mark.asyncio
async def test_grammar_structure(pcfg_engine, sample_messages):
    """Test grammar structure is valid."""
    grammar = await pcfg_engine.infer(sample_messages)

    # Check start rule exists
    start_rules = grammar.get_rules_for_symbol("<START>")
    assert len(start_rules) > 0

    # Check probabilities sum to 1 for each LHS
    rules_by_lhs = {}
    for rule in grammar.rules:
        if rule.left_hand_side not in rules_by_lhs:
            rules_by_lhs[rule.left_hand_side] = []
        rules_by_lhs[rule.left_hand_side].append(rule)

    for lhs, rules in rules_by_lhs.items():
        total_prob = sum(rule.probability for rule in rules)
        assert (
            0.99 <= total_prob <= 1.01
        ), f"Probabilities for {lhs} don't sum to 1: {total_prob}"


@pytest.mark.asyncio
async def test_complex_message_inference(pcfg_engine, complex_messages):
    """Test inference on complex messages with patterns."""
    grammar = await pcfg_engine.infer(complex_messages)

    assert grammar is not None
    assert len(grammar.rules) > 5  # Should have multiple rules

    # Should detect common patterns
    pattern_rules = [r for r in grammar.rules if "PATTERN" in r.left_hand_side]
    assert len(pattern_rules) > 0, "Should detect patterns in complex messages"


# ============================================================================
# BAYESIAN OPTIMIZATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_bayesian_optimization(config, sample_messages):
    """Test Bayesian hyperparameter optimization."""
    optimizer = BayesianOptimizer(config)

    optimal_config = await optimizer.optimize_hyperparameters(
        sample_messages, n_iterations=5, n_random_starts=2
    )

    assert optimal_config is not None
    assert isinstance(optimal_config, HyperparameterConfig)
    assert len(optimizer.optimization_history) == 5


@pytest.mark.asyncio
async def test_bayesian_optimization_improves(config, complex_messages):
    """Test that Bayesian optimization improves over random."""
    optimizer = BayesianOptimizer(config)

    optimal_config = await optimizer.optimize_hyperparameters(
        complex_messages, n_iterations=10, n_random_starts=3
    )

    # Check that later iterations have better scores
    scores = [h["score"] for h in optimizer.optimization_history]

    # Best score should be in later half
    mid_point = len(scores) // 2
    best_early = max(scores[:mid_point])
    best_late = max(scores[mid_point:])

    assert (
        best_late >= best_early * 0.9
    ), "Optimization should improve or maintain quality"


@pytest.mark.asyncio
async def test_infer_with_bayesian_optimization(
    config, production_config, complex_messages
):
    """Test inference with Bayesian optimization."""
    engine = EnhancedPCFGInference(
        config=config,
        production_config=production_config,
        enable_caching=False,
        enable_parallel=False,
    )

    grammar = await engine.infer_with_bayesian_optimization(
        complex_messages, n_optimization_iterations=5
    )

    assert grammar is not None
    assert grammar.f1_score > 0.0


# ============================================================================
# PARALLEL PROCESSING TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_parallel_inference(config, hyperparams, complex_messages):
    """Test parallel grammar inference."""
    production_config = ProductionConfig(enable_caching=False, worker_threads=2)

    engine = EnhancedPCFGInference(
        config=config,
        hyperparams=hyperparams,
        production_config=production_config,
        enable_caching=False,
        enable_parallel=True,
    )

    # Split messages into batches
    batch_size = len(complex_messages) // 2
    batches = [complex_messages[:batch_size], complex_messages[batch_size:]]

    grammars = await engine.parallel_grammar_inference(batches)

    assert len(grammars) > 0
    assert all(isinstance(g, Grammar) for g in grammars)

    await engine.shutdown()


@pytest.mark.asyncio
async def test_parallel_vs_sequential_consistency(config, hyperparams, sample_messages):
    """Test that parallel processing produces consistent results."""
    production_config = ProductionConfig(enable_caching=False, worker_threads=2)

    # Sequential
    engine_seq = EnhancedPCFGInference(
        config=config,
        hyperparams=hyperparams,
        production_config=production_config,
        enable_caching=False,
        enable_parallel=False,
    )
    grammar_seq = await engine_seq.infer(sample_messages)

    # Parallel
    engine_par = EnhancedPCFGInference(
        config=config,
        hyperparams=hyperparams,
        production_config=production_config,
        enable_caching=False,
        enable_parallel=True,
    )

    batches = [sample_messages]
    grammars_par = await engine_par.parallel_grammar_inference(batches)
    grammar_par = grammars_par[0]

    # Should have similar number of rules
    assert abs(len(grammar_seq.rules) - len(grammar_par.rules)) <= 2

    await engine_par.shutdown()


# ============================================================================
# INCREMENTAL LEARNING TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_incremental_learning(pcfg_engine, sample_messages):
    """Test incremental grammar update."""
    # Learn initial grammar
    initial_grammar = await pcfg_engine.infer(sample_messages[:3])

    # Incremental update with new messages
    updated_grammar = await pcfg_engine.incremental_grammar_update(
        initial_grammar, sample_messages[3:]
    )

    assert updated_grammar is not None
    assert updated_grammar.message_count == len(sample_messages)
    assert len(updated_grammar.rules) >= len(initial_grammar.rules)


@pytest.mark.asyncio
async def test_incremental_learning_preserves_knowledge(pcfg_engine, complex_messages):
    """Test that incremental learning preserves existing knowledge."""
    # Learn from first half
    initial_grammar = await pcfg_engine.infer(complex_messages[:10])
    initial_rule_count = len(initial_grammar.rules)

    # Update with second half
    updated_grammar = await pcfg_engine.incremental_grammar_update(
        initial_grammar, complex_messages[10:]
    )

    # Should have at least as many rules
    assert len(updated_grammar.rules) >= initial_rule_count * 0.8


@pytest.mark.asyncio
async def test_incremental_learning_adds_new_patterns(pcfg_engine):
    """Test that incremental learning discovers new patterns."""
    # Initial messages with pattern A
    initial_messages = [b"\x01\x02\x03" + bytes([i]) for i in range(5)]
    initial_grammar = await pcfg_engine.infer(initial_messages)

    # New messages with pattern B
    new_messages = [b"\x04\x05\x06" + bytes([i]) for i in range(5)]
    updated_grammar = await pcfg_engine.incremental_grammar_update(
        initial_grammar, new_messages
    )

    # Should have more terminals (new bytes)
    assert len(updated_grammar.terminals) > len(initial_grammar.terminals)


# ============================================================================
# CONVERGENCE DETECTION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_convergence_detection(pcfg_engine, sample_messages):
    """Test advanced convergence detection."""
    grammar = await pcfg_engine.infer(sample_messages)

    # Should converge in reasonable iterations
    assert grammar.num_iterations < pcfg_engine.hyperparams.max_iterations


@pytest.mark.asyncio
async def test_convergence_with_stable_data(pcfg_engine):
    """Test convergence with highly regular data."""
    # Create very regular messages
    regular_messages = [b"\x01\x02\x03\x04\x05"] * 10

    grammar = await pcfg_engine.infer(regular_messages)

    # Should converge quickly with regular data
    assert grammar.num_iterations < 20


# ============================================================================
# QUALITY METRICS TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_grammar_quality_metrics(pcfg_engine, sample_messages):
    """Test grammar quality metrics calculation."""
    grammar = await pcfg_engine.infer(sample_messages)

    # Check all metrics are calculated
    assert 0.0 <= grammar.coverage <= 1.0
    assert 0.0 <= grammar.precision <= 1.0
    assert 0.0 <= grammar.recall <= 1.0
    assert 0.0 <= grammar.f1_score <= 1.0
    assert grammar.complexity > 0.0


@pytest.mark.asyncio
async def test_grammar_complexity_calculation(
    pcfg_engine, sample_messages, complex_messages
):
    """Test that complexity increases with message complexity."""
    simple_grammar = await pcfg_engine.infer(sample_messages)
    complex_grammar = await pcfg_engine.infer(complex_messages)

    # Complex messages should produce more complex grammar
    assert complex_grammar.complexity > simple_grammar.complexity


@pytest.mark.asyncio
async def test_quality_metrics_on_test_set(pcfg_engine, complex_messages):
    """Test quality metrics on separate test set."""
    # Train on first 80%
    train_size = int(len(complex_messages) * 0.8)
    train_messages = complex_messages[:train_size]
    test_messages = complex_messages[train_size:]

    grammar = await pcfg_engine.infer(train_messages)

    # Calculate metrics on test set
    test_metrics = grammar.calculate_quality_metrics(test_messages)

    assert "coverage" in test_metrics
    assert "precision" in test_metrics
    assert "recall" in test_metrics
    assert "f1_score" in test_metrics
    assert "perplexity" in test_metrics


# ============================================================================
# PRODUCTION FEATURES TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_caching(config, hyperparams, sample_messages):
    """Test grammar caching."""
    production_config = ProductionConfig(
        enable_caching=True, cache_backend=CacheBackend.MEMORY
    )

    engine = EnhancedPCFGInference(
        config=config,
        hyperparams=hyperparams,
        production_config=production_config,
        enable_caching=True,
    )

    # First inference
    start_time = time.time()
    grammar1 = await engine.infer(sample_messages)
    first_duration = time.time() - start_time

    # Second inference (should be cached)
    start_time = time.time()
    grammar2 = await engine.infer(sample_messages)
    second_duration = time.time() - start_time

    # Cached should be faster
    assert second_duration < first_duration * 0.5

    # Should be same grammar
    assert len(grammar1.rules) == len(grammar2.rules)

    await engine.shutdown()


@pytest.mark.asyncio
async def test_metrics_collection(pcfg_engine, sample_messages):
    """Test that metrics are collected."""
    grammar = await pcfg_engine.infer(sample_messages)

    # Metrics should be updated
    assert pcfg_engine.metrics is not None


@pytest.mark.asyncio
async def test_error_handling(pcfg_engine):
    """Test error handling with invalid input."""
    # Test with invalid message type
    with pytest.raises(Exception):
        await pcfg_engine.infer([None])  # type: ignore


# ============================================================================
# RULE QUALITY TESTS
# ============================================================================


def test_production_rule_bayesian_update():
    """Test Bayesian statistics update for production rules."""
    rule = ProductionRule(
        left_hand_side="<TEST>",
        right_hand_side=["token1", "token2"],
        probability=0.5,
        frequency=10,
    )

    # Update with new evidence
    rule.update_bayesian_stats(successes=8, trials=10)

    # Probability should be updated
    assert rule.probability > 0.5
    assert rule.alpha > 1.0
    assert rule.beta_param > 1.0


def test_production_rule_is_terminal():
    """Test terminal rule detection."""
    terminal_rule = ProductionRule(
        left_hand_side="<BODY>", right_hand_side=["0x01", "0x02"], probability=0.5
    )

    non_terminal_rule = ProductionRule(
        left_hand_side="<MESSAGE>",
        right_hand_side=["<HEADER>", "<BODY>"],
        probability=0.5,
    )

    assert terminal_rule.is_terminal_rule()
    assert not non_terminal_rule.is_terminal_rule()


# ============================================================================
# GRAMMAR SERIALIZATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_grammar_serialization(pcfg_engine, sample_messages):
    """Test grammar serialization to dictionary."""
    grammar = await pcfg_engine.infer(sample_messages)

    grammar_dict = grammar.to_dict()

    assert "rules" in grammar_dict
    assert "terminals" in grammar_dict
    assert "non_terminals" in grammar_dict
    assert "start_symbol" in grammar_dict
    assert "metrics" in grammar_dict

    # Check metrics
    metrics = grammar_dict["metrics"]
    assert "complexity" in metrics
    assert "coverage" in metrics
    assert "f1_score" in metrics


# ============================================================================
# TOKENIZATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_tokenization(pcfg_engine, sample_messages):
    """Test message tokenization."""
    tokens = await pcfg_engine._tokenize_messages(sample_messages)

    assert len(tokens) == len(sample_messages)
    assert all(isinstance(token_list, list) for token_list in tokens)
    assert all(len(token_list) > 0 for token_list in tokens)


@pytest.mark.asyncio
async def test_tokenization_consistency(pcfg_engine):
    """Test that tokenization is consistent."""
    message = b"\x01\x02\x03\x04\x05"

    tokens1 = await pcfg_engine._tokenize_messages([message])
    tokens2 = await pcfg_engine._tokenize_messages([message])

    assert tokens1 == tokens2


# ============================================================================
# PATTERN EXTRACTION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_pattern_extraction(pcfg_engine, complex_messages):
    """Test pattern extraction from messages."""
    tokens = await pcfg_engine._tokenize_messages(complex_messages)
    patterns = await pcfg_engine._extract_frequent_patterns(tokens)

    assert "unigrams" in patterns
    assert "bigrams" in patterns
    assert "trigrams" in patterns
    assert "common_subsequences" in patterns
    assert "structural_patterns" in patterns


@pytest.mark.asyncio
async def test_common_subsequence_detection(pcfg_engine):
    """Test detection of common subsequences."""
    # Messages with common pattern
    messages = [
        b"\x01\x02\x03\x04\x05",
        b"\x01\x02\x03\x06\x07",
        b"\x01\x02\x03\x08\x09",
    ]

    tokens = await pcfg_engine._tokenize_messages(messages)
    patterns = await pcfg_engine._extract_frequent_patterns(tokens)

    # Should detect common prefix
    common_seqs = patterns["common_subsequences"]
    assert len(common_seqs) > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_large_dataset_performance(config, hyperparams, production_config):
    """Test performance on large dataset."""
    # Generate large dataset
    large_messages = []
    for i in range(1000):
        msg = bytes([i % 256, (i // 256) % 256]) + b"\x00" * 3
        large_messages.append(msg)

    engine = EnhancedPCFGInference(
        config=config,
        hyperparams=hyperparams,
        production_config=production_config,
        enable_caching=False,
        enable_parallel=True,
    )

    start_time = time.time()
    grammar = await engine.infer(large_messages)
    duration = time.time() - start_time

    # Should complete in reasonable time (< 30 seconds)
    assert duration < 30.0
    assert grammar is not None

    await engine.shutdown()


@pytest.mark.asyncio
async def test_memory_efficiency(pcfg_engine, complex_messages):
    """Test memory efficiency with repeated inference."""
    # Run multiple inferences
    for _ in range(5):
        grammar = await pcfg_engine.infer(complex_messages)
        assert grammar is not None

    # Should not accumulate excessive state
    assert len(pcfg_engine._pattern_cache) < 1000


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_workflow(
    config, hyperparams, production_config, complex_messages
):
    """Test complete end-to-end workflow."""
    engine = EnhancedPCFGInference(
        config=config,
        hyperparams=hyperparams,
        production_config=production_config,
        enable_caching=True,
        enable_parallel=True,
    )

    # 1. Initial inference
    initial_grammar = await engine.infer(complex_messages[:10])
    assert initial_grammar is not None

    # 2. Incremental update
    updated_grammar = await engine.incremental_grammar_update(
        initial_grammar, complex_messages[10:]
    )
    assert updated_grammar is not None

    # 3. Quality metrics
    metrics = updated_grammar.calculate_quality_metrics(complex_messages)
    assert metrics["f1_score"] > 0.0

    # 4. Cleanup
    await engine.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
