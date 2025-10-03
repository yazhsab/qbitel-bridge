"""Field detector training and data loader tests."""

import importlib

import pytest

torch = pytest.importorskip("torch")

torchcrf = None
for module_name in ("torchcrf", "TorchCRF"):
    try:
        torchcrf = importlib.import_module(module_name)
        break
    except ModuleNotFoundError:
        continue

if torchcrf is None:
    pytest.skip("torchcrf is required for field detector tests")

from ai_engine.core.config import Config
from ai_engine.detection.field_detector import FieldDetector


@pytest.mark.asyncio
async def test_field_detector_training_loop_runs():
    """Ensure the training and validation loops execute with real data loaders."""
    config = Config()
    detector = FieldDetector(config)

    # Reduce model size for faster test execution
    detector.embedding_dim = 16
    detector.hidden_dim = 32
    detector.num_layers = 1
    detector.max_sequence_length = 32

    await detector.initialize()

    training_samples = [
        (b"HEADERBODY", [(0, 6, "field")]),
        (b"PAYLOAD123", [(0, 4, "field")]),
    ]

    history = detector.train(
        training_data=training_samples,
        validation_data=training_samples,
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-2,
    )

    assert history["train_loss"], "Training loss history should not be empty"
    assert history["val_loss"], "Validation loss history should not be empty"
    assert history["val_f1"], "Validation F1 history should not be empty"
    assert history["train_loss"][0] >= 0
