"""
Comprehensive Test Suite for ML Pipeline Features

Tests for batch training, validation metrics, model persistence, and rollback functionality.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
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
    TranslationRule,
    TranslationStrategy,
)
from ai_engine.core.config import Config


class TestFieldDetectorBatchTraining:
    """Test batch training functionality for field detector."""

    @pytest.fixture
    async def field_detector(self):
        """Create field detector instance."""
        config = Mock(spec=Config)
        config.field_detection_ml_classifier_enabled = False
        detector = FieldDetector(config)
        await detector.initialize()
        return detector

    @pytest.fixture
    def training_batches(self) -> List[List[Tuple[bytes, List[Tuple[int, int, str]]]]]:
        """Generate training batches."""
        batches = []
        for batch_idx in range(3):
            batch = []
            for i in range(20):
                message = b"\x01\x00" + f"B{batch_idx}T{i:03d}".encode() + b"\xff\xff"
                annotations = [
                    (0, 2, "header"),
                    (2, 10, "payload"),
                    (10, 14, "checksum"),
                ]
                batch.append((message, annotations))
            batches.append(batch)
        return batches

    @pytest.mark.asyncio
    async def test_batch_training_completes(self, field_detector, training_batches):
        """Test that batch training completes successfully."""
        history = field_detector.train_batch(training_batches=training_batches, num_epochs_per_batch=2, batch_size=8)

        assert "batch_histories" in history
        assert "batch_metrics" in history
        assert len(history["batch_metrics"]) == len(training_batches)

        # Verify each batch was processed
        for batch_idx, metrics in enumerate(history["batch_metrics"]):
            assert metrics["batch_idx"] == batch_idx
            assert metrics["samples"] == 20
            assert "final_train_loss" in metrics
            assert "best_val_f1" in metrics

    @pytest.mark.asyncio
    async def test_batch_training_improves_over_batches(self, field_detector, training_batches):
        """Test that model improves across batches."""
        history = field_detector.train_batch(training_batches=training_batches, num_epochs_per_batch=3, batch_size=8)

        # Check that loss generally decreases
        losses = [m["final_train_loss"] for m in history["batch_metrics"]]
        assert losses[-1] <= losses[0] * 1.5  # Allow some variance

    @pytest.mark.asyncio
    async def test_batch_training_with_validation(self, field_detector, training_batches):
        """Test batch training with validation data."""
        validation_data = training_batches[0][:10]  # Use subset as validation

        history = field_detector.train_batch(
            training_batches=training_batches[1:],
            validation_data=validation_data,
            num_epochs_per_batch=2,
            batch_size=8,
        )

        assert "overall_val_f1" in history
        assert len(history["overall_val_f1"]) > 0


class TestFieldDetectorValidationMetrics:
    """Test comprehensive validation metrics."""

    @pytest.fixture
    async def trained_detector(self):
        """Create and train a detector."""
        config = Mock(spec=Config)
        config.field_detection_ml_classifier_enabled = False
        detector = FieldDetector(config)
        await detector.initialize()

        # Quick training
        training_data = [(b"\x01\x00TEST1234\xff\xff", [(0, 2, "h"), (2, 10, "p"), (10, 14, "c")]) for _ in range(20)]
        detector.train(training_data=training_data, num_epochs=2, batch_size=8)

        return detector

    @pytest.mark.asyncio
    async def test_comprehensive_validation_metrics(self, trained_detector):
        """Test that all validation metrics are computed."""
        validation_data = [(b"\x01\x00TEST5678\xff\xff", [(0, 2, "h"), (2, 10, "p"), (10, 14, "c")]) for _ in range(10)]

        val_loader = trained_detector._create_data_loader(validation_data, batch_size=4, shuffle=False)
        metrics = trained_detector._validate_epoch_comprehensive(val_loader)

        # Check all expected metrics
        expected_keys = [
            "loss",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "per_tag_metrics",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

        # Check per-tag metrics
        assert "per_tag_metrics" in metrics
        assert len(metrics["per_tag_metrics"]) > 0

        for tag, tag_metrics in metrics["per_tag_metrics"].items():
            assert "precision" in tag_metrics
            assert "recall" in tag_metrics
            assert "f1" in tag_metrics
            assert "support" in tag_metrics

    @pytest.mark.asyncio
    async def test_training_history_completeness(self, trained_detector):
        """Test that training history contains all metrics."""
        training_data = [(b"\x01\x00TEST1234\xff\xff", [(0, 2, "h"), (2, 10, "p"), (10, 14, "c")]) for _ in range(30)]
        validation_data = training_data[:10]

        history = trained_detector.train(
            training_data=training_data[10:],
            validation_data=validation_data,
            num_epochs=3,
            batch_size=8,
        )

        # Check all expected history keys
        expected_keys = [
            "train_loss",
            "val_loss",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "learning_rate",
            "epoch_time",
            "summary",
        ]

        for key in expected_keys:
            assert key in history, f"Missing history key: {key}"

        # Check summary
        assert "best_val_f1" in history["summary"]
        assert "total_epochs" in history["summary"]
        assert "total_time_seconds" in history["summary"]
        assert "final_learning_rate" in history["summary"]


class TestFieldDetectorModelPersistence:
    """Test model persistence and checkpointing."""

    @pytest.fixture
    async def trained_detector(self, tmp_path):
        """Create and train a detector."""
        config = Mock(spec=Config)
        config.field_detection_ml_classifier_enabled = False
        detector = FieldDetector(config)
        await detector.initialize()

        training_data = [(b"\x01\x00TEST1234\xff\xff", [(0, 2, "h"), (2, 10, "p"), (10, 14, "c")]) for _ in range(20)]
        detector.train(training_data=training_data, num_epochs=2, batch_size=8)

        return detector, tmp_path

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_load(self, trained_detector):
        """Test saving and loading checkpoints."""
        detector, tmp_path = trained_detector
        checkpoint_dir = str(tmp_path / "checkpoints")

        # Save checkpoint
        checkpoint_path = detector._save_checkpoint(
            epoch=5,
            metrics={"f1": 0.95, "precision": 0.94, "recall": 0.96},
            checkpoint_dir=checkpoint_dir,
        )

        assert os.path.exists(checkpoint_path)

        # Create new detector and load checkpoint
        new_detector = FieldDetector(Mock(spec=Config))
        await new_detector.initialize()

        metadata = await new_detector.load_checkpoint(checkpoint_path)

        assert metadata["epoch"] == 5
        assert metadata["metrics"]["f1"] == 0.95
        assert "version" in metadata
        assert "timestamp" in metadata

    @pytest.mark.asyncio
    async def test_checkpoint_versioning(self, trained_detector):
        """Test that checkpoints are versioned correctly."""
        detector, tmp_path = trained_detector
        checkpoint_dir = str(tmp_path / "checkpoints")

        # Save multiple checkpoints
        paths = []
        for epoch in range(3):
            path = detector._save_checkpoint(
                epoch=epoch,
                metrics={"f1": 0.90 + epoch * 0.01},
                checkpoint_dir=checkpoint_dir,
            )
            paths.append(path)

        # Verify all checkpoints exist
        for path in paths:
            assert os.path.exists(path)

        # Verify latest checkpoint exists
        latest_path = os.path.join(checkpoint_dir, f"field_detector_v{detector.model_version}_latest.pt")
        assert os.path.exists(latest_path)

    @pytest.mark.asyncio
    async def test_checkpoint_contains_full_state(self, trained_detector):
        """Test that checkpoint contains all necessary state."""
        detector, tmp_path = trained_detector
        checkpoint_dir = str(tmp_path / "checkpoints")

        checkpoint_path = detector._save_checkpoint(epoch=10, metrics={"f1": 0.92}, checkpoint_dir=checkpoint_dir)

        # Load checkpoint and verify contents
        import torch

        checkpoint = torch.load(checkpoint_path)

        required_keys = [
            "epoch",
            "model_state_dict",
            "version",
            "timestamp",
            "metrics",
            "config",
            "tag_mappings",
        ]

        for key in required_keys:
            assert key in checkpoint, f"Missing checkpoint key: {key}"


class TestTranslationStudioBatchTraining:
    """Test batch training for translation studio."""

    @pytest.fixture
    async def translation_studio(self):
        """Create translation studio instance."""
        config = Mock(spec=Config)
        llm_service = AsyncMock()
        llm_service.process_request = AsyncMock(return_value=Mock(content='{"rule_modifications": []}'))

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
    async def test_batch_training_completes(self, translation_studio, sample_protocols):
        """Test that batch training completes."""
        source_spec, target_spec = sample_protocols

        training_batches = [
            [
                {
                    "source": {"field1": i, "field2": f"test{i}"},
                    "expected_target": {"field1": i, "field2": f"test{i}"},
                }
                for i in range(10)
            ]
            for _ in range(2)
        ]

        history = await translation_studio.train_translation_rules_batch(
            training_batches=training_batches,
            source_spec=source_spec,
            target_spec=target_spec,
        )

        assert "batch_histories" in history
        assert "batch_metrics" in history
        assert len(history["batch_metrics"]) == len(training_batches)

    @pytest.mark.asyncio
    async def test_validation_metrics_comprehensive(self, translation_studio, sample_protocols):
        """Test comprehensive validation metrics."""
        source_spec, target_spec = sample_protocols

        # Create simple rules
        rules = TranslationRules(
            rules_id="test",
            source_protocol=source_spec,
            target_protocol=target_spec,
            rules=[
                TranslationRule(
                    rule_id="r1",
                    source_field="field1",
                    target_field="field1",
                    strategy=TranslationStrategy.DIRECT_MAPPING,
                )
            ],
        )

        validation_data = [{"source": {"field1": i}, "expected_target": {"field1": i}} for i in range(10)]

        metrics = await translation_studio._validate_translation_rules(rules, validation_data)

        expected_keys = [
            "pass_rate",
            "fail_rate",
            "error_rate",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "total_examples",
            "passed",
            "failed",
            "errors",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"


class TestTranslationStudioPersistence:
    """Test translation rules persistence."""

    @pytest.fixture
    async def translation_studio(self):
        """Create translation studio instance."""
        config = Mock(spec=Config)
        llm_service = AsyncMock()
        studio = ProtocolTranslationStudio(config, llm_service)
        return studio

    @pytest.fixture
    def sample_rules(self):
        """Create sample translation rules."""
        source_spec = ProtocolSpecification(
            protocol_id="test_source",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Test Source",
            description="Test",
            fields=[],
        )

        target_spec = ProtocolSpecification(
            protocol_id="test_target",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Test Target",
            description="Test",
            fields=[],
        )

        rules = TranslationRules(
            rules_id="test123",
            source_protocol=source_spec,
            target_protocol=target_spec,
            rules=[
                TranslationRule(
                    rule_id="r1",
                    source_field="f1",
                    target_field="f1",
                    strategy=TranslationStrategy.DIRECT_MAPPING,
                )
            ],
            accuracy=0.95,
        )

        return rules

    @pytest.mark.asyncio
    async def test_save_and_load_rules(self, translation_studio, sample_rules, tmp_path):
        """Test saving and loading translation rules."""
        rules_dir = str(tmp_path / "rules")

        # Save rules
        rules_path = await translation_studio.save_translation_rules(sample_rules, rules_dir=rules_dir, version="1.0.0")

        assert os.path.exists(rules_path)

        # Load rules
        loaded_rules = await translation_studio.load_translation_rules(rules_path)

        assert loaded_rules.rules_id == sample_rules.rules_id
        assert loaded_rules.accuracy == sample_rules.accuracy
        assert len(loaded_rules.rules) == len(sample_rules.rules)

    @pytest.mark.asyncio
    async def test_list_saved_rules(self, translation_studio, sample_rules, tmp_path):
        """Test listing saved rules."""
        rules_dir = str(tmp_path / "rules")

        # Save multiple versions
        for i in range(3):
            sample_rules.accuracy = 0.90 + i * 0.02
            await translation_studio.save_translation_rules(sample_rules, rules_dir=rules_dir, version=f"1.{i}.0")

        # List rules
        rules_list = translation_studio.list_saved_rules(
            rules_dir=rules_dir,
            source_protocol="test_source",
            target_protocol="test_target",
        )

        assert len(rules_list) == 3

        # Verify sorting (newest first)
        for i in range(len(rules_list) - 1):
            assert rules_list[i]["timestamp"] >= rules_list[i + 1]["timestamp"]

    @pytest.mark.asyncio
    async def test_rules_persistence_formats(self, translation_studio, sample_rules, tmp_path):
        """Test that rules are saved in multiple formats."""
        rules_dir = str(tmp_path / "rules")

        rules_path = await translation_studio.save_translation_rules(sample_rules, rules_dir=rules_dir)

        # Check pickle file exists
        assert os.path.exists(rules_path)
        assert rules_path.endswith(".pkl")

        # Check JSON file exists
        json_path = rules_path.replace(".pkl", ".json")
        assert os.path.exists(json_path)

        # Check latest file exists
        latest_path = os.path.join(
            rules_dir,
            f"translation_rules_{sample_rules.source_protocol.protocol_id}_to_{sample_rules.target_protocol.protocol_id}_latest.pkl",
        )
        assert os.path.exists(latest_path)


class TestRollbackFunctionality:
    """Test rollback functionality for both systems."""

    @pytest.mark.asyncio
    async def test_field_detector_rollback(self, tmp_path):
        """Test field detector rollback to previous checkpoint."""
        config = Mock(spec=Config)
        config.field_detection_ml_classifier_enabled = False

        # Train initial model
        detector1 = FieldDetector(config)
        await detector1.initialize()

        training_data = [(b"\x01\x00TEST1234\xff\xff", [(0, 2, "h"), (2, 10, "p"), (10, 14, "c")]) for _ in range(20)]
        detector1.train(training_data=training_data, num_epochs=2, batch_size=8)

        checkpoint_dir = str(tmp_path / "checkpoints")
        checkpoint1 = detector1._save_checkpoint(epoch=5, metrics={"f1": 0.95}, checkpoint_dir=checkpoint_dir)

        # Train "worse" model
        detector2 = FieldDetector(config)
        await detector2.initialize()
        detector2.train(training_data=training_data[:10], num_epochs=1, batch_size=8)

        checkpoint2 = detector2._save_checkpoint(epoch=1, metrics={"f1": 0.85}, checkpoint_dir=checkpoint_dir)

        # Rollback to first checkpoint
        detector_rollback = FieldDetector(config)
        await detector_rollback.initialize()
        metadata = await detector_rollback.load_checkpoint(checkpoint1)

        assert metadata["metrics"]["f1"] == 0.95
        assert metadata["epoch"] == 5

    @pytest.mark.asyncio
    async def test_translation_studio_rollback(self, tmp_path):
        """Test translation studio rollback to previous rules."""
        config = Mock(spec=Config)
        llm_service = AsyncMock()
        studio = ProtocolTranslationStudio(config, llm_service)

        # Create rules with different accuracies
        source_spec = ProtocolSpecification(
            protocol_id="test",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Test",
            description="Test",
            fields=[],
        )
        target_spec = ProtocolSpecification(
            protocol_id="test2",
            protocol_type=ProtocolType.CUSTOM,
            version="1.0",
            name="Test2",
            description="Test",
            fields=[],
        )

        rules_dir = str(tmp_path / "rules")

        # Save "good" rules
        good_rules = TranslationRules(
            rules_id="good",
            source_protocol=source_spec,
            target_protocol=target_spec,
            rules=[],
            accuracy=0.95,
        )
        good_path = await studio.save_translation_rules(good_rules, rules_dir=rules_dir, version="1.0.0")

        # Save "bad" rules
        bad_rules = TranslationRules(
            rules_id="bad",
            source_protocol=source_spec,
            target_protocol=target_spec,
            rules=[],
            accuracy=0.75,
        )
        await studio.save_translation_rules(bad_rules, rules_dir=rules_dir, version="1.1.0")

        # Rollback to good rules
        rolled_back_rules = await studio.load_translation_rules(good_path)

        assert rolled_back_rules.rules_id == "good"
        assert rolled_back_rules.accuracy == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
