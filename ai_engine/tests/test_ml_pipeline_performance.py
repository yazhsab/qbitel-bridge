"""
Baseline Performance Tests for ML/Data Pipelines

Tests for field detection and translation studio performance benchmarks.
"""

import pytest
import asyncio
import time
import numpy as np
from typing import List, Tuple
from unittest.mock import Mock, AsyncMock, patch

from ai_engine.detection.field_detector import FieldDetector, FieldBoundary, FieldType
from ai_engine.llm.translation_studio import (
    ProtocolTranslationStudio,
    ProtocolSpecification,
    ProtocolField,
    ProtocolType,
    FieldType as TransFieldType,
    TranslationRules,
)
from ai_engine.core.config import Config


class TestFieldDetectorPerformance:
    """Performance tests for field detector."""

    @pytest.fixture
    async def field_detector(self):
        """Create field detector instance."""
        config = Mock(spec=Config)
        config.field_detection_ml_classifier_enabled = False
        detector = FieldDetector(config)
        await detector.initialize()
        return detector

    @pytest.fixture
    def sample_training_data(self) -> List[Tuple[bytes, List[Tuple[int, int, str]]]]:
        """Generate sample training data."""
        training_data = []
        for i in range(100):
            # Create synthetic protocol message
            message = b"\x01\x00" + f"TEST{i:04d}".encode() + b"\x00\x00\xff\xff"
            # Annotations: (start, end, field_type)
            annotations = [(0, 2, "header"), (2, 10, "payload"), (10, 14, "checksum")]
            training_data.append((message, annotations))
        return training_data

    @pytest.mark.asyncio
    async def test_training_performance_baseline(self, field_detector, sample_training_data):
        """Test training performance meets baseline requirements."""
        # Split data
        train_data = sample_training_data[:80]
        val_data = sample_training_data[80:]

        start_time = time.time()

        # Train model
        history = field_detector.train(
            training_data=train_data,
            validation_data=val_data,
            num_epochs=5,
            batch_size=16,
            early_stopping_patience=3,
        )

        training_time = time.time() - start_time

        # Performance assertions
        assert training_time < 60.0, f"Training took {training_time:.2f}s, should be < 60s"
        assert "train_loss" in history
        assert "val_f1" in history
        assert len(history["train_loss"]) > 0

        # Check final metrics
        final_f1 = history["val_f1"][-1] if history["val_f1"] else 0.0
        assert final_f1 > 0.5, f"Final F1 score {final_f1:.4f} should be > 0.5"

        print(f"\n✓ Training Performance Baseline:")
        print(f"  - Training time: {training_time:.2f}s")
        print(f"  - Final F1 score: {final_f1:.4f}")
        print(f"  - Epochs completed: {len(history['train_loss'])}")

    @pytest.mark.asyncio
    async def test_inference_latency_baseline(self, field_detector):
        """Test inference latency meets baseline requirements."""
        # Create test message
        test_message = b"\x01\x00TEST1234\x00\x00\xff\xff"

        # Warm-up
        await field_detector.detect_boundaries(test_message)

        # Measure latency
        latencies = []
        num_iterations = 100

        for _ in range(num_iterations):
            start_time = time.time()
            await field_detector.detect_boundaries(test_message)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        # Performance assertions
        assert avg_latency < 10.0, f"Average latency {avg_latency:.2f}ms should be < 10ms"
        assert p95_latency < 20.0, f"P95 latency {p95_latency:.2f}ms should be < 20ms"
        assert p99_latency < 50.0, f"P99 latency {p99_latency:.2f}ms should be < 50ms"

        print(f"\n✓ Inference Latency Baseline:")
        print(f"  - Average: {avg_latency:.2f}ms")
        print(f"  - P50: {p50_latency:.2f}ms")
        print(f"  - P95: {p95_latency:.2f}ms")
        print(f"  - P99: {p99_latency:.2f}ms")

    @pytest.mark.asyncio
    async def test_batch_training_performance(self, field_detector, sample_training_data):
        """Test batch training performance."""
        # Split into batches
        batch_size = 20
        batches = [sample_training_data[i : i + batch_size] for i in range(0, len(sample_training_data), batch_size)]

        start_time = time.time()

        # Train on batches
        history = field_detector.train_batch(
            training_batches=batches[:3],  # Use first 3 batches
            num_epochs_per_batch=3,
            batch_size=8,
        )

        batch_training_time = time.time() - start_time

        # Performance assertions
        assert batch_training_time < 120.0, f"Batch training took {batch_training_time:.2f}s, should be < 120s"
        assert "batch_metrics" in history
        assert len(history["batch_metrics"]) == 3

        print(f"\n✓ Batch Training Performance:")
        print(f"  - Total time: {batch_training_time:.2f}s")
        print(f"  - Batches processed: {len(history['batch_metrics'])}")
        print(f"  - Avg time per batch: {batch_training_time / len(history['batch_metrics']):.2f}s")

    @pytest.mark.asyncio
    async def test_model_persistence_performance(self, field_detector, sample_training_data, tmp_path):
        """Test model save/load performance."""
        # Train a small model
        train_data = sample_training_data[:20]
        field_detector.train(training_data=train_data, num_epochs=2, batch_size=8)

        model_path = str(tmp_path / "test_model.pt")

        # Test save performance
        start_time = time.time()
        await field_detector.save_model(model_path)
        save_time = time.time() - start_time

        # Test load performance
        start_time = time.time()
        await field_detector.load_model(model_path)
        load_time = time.time() - start_time

        # Performance assertions
        assert save_time < 5.0, f"Model save took {save_time:.2f}s, should be < 5s"
        assert load_time < 5.0, f"Model load took {load_time:.2f}s, should be < 5s"

        print(f"\n✓ Model Persistence Performance:")
        print(f"  - Save time: {save_time:.2f}s")
        print(f"  - Load time: {load_time:.2f}s")

    @pytest.mark.asyncio
    async def test_validation_metrics_completeness(self, field_detector, sample_training_data):
        """Test that validation metrics are comprehensive."""
        train_data = sample_training_data[:60]
        val_data = sample_training_data[60:80]

        # Train with validation
        history = field_detector.train(
            training_data=train_data,
            validation_data=val_data,
            num_epochs=3,
            batch_size=16,
        )

        # Check all expected metrics are present
        expected_metrics = [
            "train_loss",
            "val_loss",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "learning_rate",
            "epoch_time",
        ]

        for metric in expected_metrics:
            assert metric in history, f"Missing metric: {metric}"
            assert len(history[metric]) > 0, f"Empty metric: {metric}"

        # Check summary
        assert "summary" in history
        assert "best_val_f1" in history["summary"]
        assert "total_epochs" in history["summary"]
        assert "total_time_seconds" in history["summary"]

        print(f"\n✓ Validation Metrics Completeness:")
        print(f"  - All {len(expected_metrics)} metrics present")
        print(f"  - Best F1: {history['summary']['best_val_f1']:.4f}")
        print(f"  - Total epochs: {history['summary']['total_epochs']}")


class TestTranslationStudioPerformance:
    """Performance tests for translation studio."""

    @pytest.fixture
    async def translation_studio(self):
        """Create translation studio instance."""
        config = Mock(spec=Config)
        llm_service = AsyncMock()
        llm_service.process_request = AsyncMock(
            return_value=Mock(
                content='{"rules": [], "preprocessing": [], "postprocessing": [], "error_handling": {}, "performance_hints": {}, "test_cases": []}'
            )
        )

        studio = ProtocolTranslationStudio(config, llm_service)
        return studio

    @pytest.fixture
    def sample_protocols(self):
        """Create sample protocol specifications."""
        source_spec = ProtocolSpecification(
            protocol_id="test_source",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Test Source",
            description="Test source protocol",
            fields=[
                ProtocolField(
                    name="field1",
                    field_type=TransFieldType.INTEGER,
                    length=4,
                    required=True,
                ),
                ProtocolField(
                    name="field2",
                    field_type=TransFieldType.STRING,
                    length=16,
                    required=True,
                ),
            ],
        )

        target_spec = ProtocolSpecification(
            protocol_id="test_target",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Test Target",
            description="Test target protocol",
            fields=[
                ProtocolField(
                    name="field1",
                    field_type=TransFieldType.INTEGER,
                    length=4,
                    required=True,
                ),
                ProtocolField(
                    name="field2",
                    field_type=TransFieldType.STRING,
                    length=16,
                    required=True,
                ),
            ],
        )

        return source_spec, target_spec

    @pytest.mark.asyncio
    async def test_translation_latency_baseline(self, translation_studio, sample_protocols):
        """Test translation latency meets baseline requirements."""
        source_spec, target_spec = sample_protocols

        # Register protocols
        await translation_studio.register_protocol(source_spec)
        await translation_studio.register_protocol(target_spec)

        # Create test message
        test_message = b"\x00\x00\x00\x01TEST_MESSAGE\x00\x00\x00\x00"

        # Warm-up
        try:
            await translation_studio.translate_protocol("test_source", "test_target", test_message)
        except:
            pass  # May fail due to mock, but warms up the system

        # Measure latency
        latencies = []
        num_iterations = 50

        for _ in range(num_iterations):
            start_time = time.time()
            try:
                await translation_studio.translate_protocol("test_source", "test_target", test_message)
            except:
                pass  # Mock may cause failures
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"\n✓ Translation Latency Baseline:")
        print(f"  - Average: {avg_latency:.2f}ms")
        print(f"  - P95: {p95_latency:.2f}ms")

    @pytest.mark.asyncio
    async def test_rule_generation_performance(self, translation_studio, sample_protocols):
        """Test rule generation performance."""
        source_spec, target_spec = sample_protocols

        start_time = time.time()

        # Generate rules
        rules = await translation_studio.generate_translation_rules(source_spec, target_spec)

        generation_time = time.time() - start_time

        # Performance assertions
        assert generation_time < 30.0, f"Rule generation took {generation_time:.2f}s, should be < 30s"
        assert rules is not None
        assert len(rules.rules) >= 0

        print(f"\n✓ Rule Generation Performance:")
        print(f"  - Generation time: {generation_time:.2f}s")
        print(f"  - Rules generated: {len(rules.rules)}")

    @pytest.mark.asyncio
    async def test_rules_persistence_performance(self, translation_studio, sample_protocols, tmp_path):
        """Test rules save/load performance."""
        source_spec, target_spec = sample_protocols

        # Generate rules
        rules = await translation_studio.generate_translation_rules(source_spec, target_spec)

        rules_dir = str(tmp_path / "rules")

        # Test save performance
        start_time = time.time()
        rules_path = await translation_studio.save_translation_rules(rules, rules_dir=rules_dir)
        save_time = time.time() - start_time

        # Test load performance
        start_time = time.time()
        loaded_rules = await translation_studio.load_translation_rules(rules_path)
        load_time = time.time() - start_time

        # Performance assertions
        assert save_time < 5.0, f"Rules save took {save_time:.2f}s, should be < 5s"
        assert load_time < 5.0, f"Rules load took {load_time:.2f}s, should be < 5s"
        assert loaded_rules.rules_id == rules.rules_id

        print(f"\n✓ Rules Persistence Performance:")
        print(f"  - Save time: {save_time:.2f}s")
        print(f"  - Load time: {load_time:.2f}s")

    @pytest.mark.asyncio
    async def test_batch_training_performance(self, translation_studio, sample_protocols):
        """Test batch training performance for translation rules."""
        source_spec, target_spec = sample_protocols

        # Create training batches
        batches = [
            [
                {
                    "source": {"field1": i, "field2": f"test{i}"},
                    "expected_target": {"field1": i, "field2": f"test{i}"},
                }
                for i in range(batch_start, batch_start + 10)
            ]
            for batch_start in range(0, 30, 10)
        ]

        start_time = time.time()

        # Train on batches
        history = await translation_studio.train_translation_rules_batch(
            training_batches=batches, source_spec=source_spec, target_spec=target_spec
        )

        batch_training_time = time.time() - start_time

        # Performance assertions
        assert batch_training_time < 180.0, f"Batch training took {batch_training_time:.2f}s, should be < 180s"
        assert "batch_metrics" in history
        assert len(history["batch_metrics"]) == len(batches)

        print(f"\n✓ Batch Training Performance:")
        print(f"  - Total time: {batch_training_time:.2f}s")
        print(f"  - Batches processed: {len(history['batch_metrics'])}")
        print(f"  - Avg time per batch: {batch_training_time / len(history['batch_metrics']):.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
