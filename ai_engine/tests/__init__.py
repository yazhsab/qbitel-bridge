"""
CRONOS AI Engine - Test Suite

This module provides comprehensive testing for all AI Engine components.
"""

import sys
import os
from pathlib import Path

# Add ai_engine to path for testing
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Ensure the ai_engine package itself is available before relative imports
ai_engine_path = project_root / "ai_engine"
if str(ai_engine_path) not in sys.path:
    sys.path.insert(0, str(ai_engine_path))

# Test configuration
TEST_DATA_PATH = Path(__file__).parent / "data"
TEST_MODELS_PATH = Path(__file__).parent / "models"
TEST_OUTPUT_PATH = Path(__file__).parent / "output"

# Ensure test directories exist
TEST_DATA_PATH.mkdir(exist_ok=True)
TEST_MODELS_PATH.mkdir(exist_ok=True)
TEST_OUTPUT_PATH.mkdir(exist_ok=True)


# Common test utilities
class TestConfig:
    """Test configuration settings."""

    # Test data settings
    SAMPLE_PROTOCOL_DATA = {
        "http_request": b"GET /api/v1/users HTTP/1.1\r\nHost: example.com\r\nUser-Agent: TestAgent/1.0\r\n\r\n",
        "modbus_frame": bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x02, 0xC4, 0x0B]),
        "hl7_message": b"MSH|^~\\&|SENDER|HOSPITAL|RECEIVER|CLINIC|20230101120000||ADT^A08|123456|P|2.5\r",
        "iso8583_message": bytes(
            [0x00, 0x21, 0x01, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        ),
        "random_data": os.urandom(256),
    }

    # Test thresholds
    CONFIDENCE_THRESHOLD = 0.7
    ANOMALY_THRESHOLD = 0.5
    PERFORMANCE_THRESHOLD_MS = 1000

    # Model test settings
    MIN_ACCURACY = 0.8
    MIN_PRECISION = 0.75
    MIN_RECALL = 0.75
    MIN_F1_SCORE = 0.75


def create_test_data():
    """Create sample test data files."""
    import json
    import pickle

    # Create sample training data
    training_data = {
        "protocol_samples": [
            {
                "data": TestConfig.SAMPLE_PROTOCOL_DATA["http_request"].hex(),
                "label": "http",
            },
            {
                "data": TestConfig.SAMPLE_PROTOCOL_DATA["modbus_frame"].hex(),
                "label": "modbus",
            },
            {
                "data": TestConfig.SAMPLE_PROTOCOL_DATA["hl7_message"].hex(),
                "label": "hl7",
            },
            {
                "data": TestConfig.SAMPLE_PROTOCOL_DATA["iso8583_message"].hex(),
                "label": "iso8583",
            },
        ]
    }

    with open(TEST_DATA_PATH / "training_samples.json", "w") as f:
        json.dump(training_data, f, indent=2)

    # Create sample field detection data
    field_data = {
        "http_fields": [
            {"start": 0, "end": 3, "type": "method", "value": "GET"},
            {"start": 4, "end": 17, "type": "path", "value": "/api/v1/users"},
            {"start": 18, "end": 26, "type": "version", "value": "HTTP/1.1"},
        ]
    }

    with open(TEST_DATA_PATH / "field_samples.json", "w") as f:
        json.dump(field_data, f, indent=2)

    # Create sample anomaly data
    anomaly_data = {
        "normal_samples": [TestConfig.SAMPLE_PROTOCOL_DATA["http_request"].hex()],
        "anomalous_samples": [TestConfig.SAMPLE_PROTOCOL_DATA["random_data"].hex()],
    }

    with open(TEST_DATA_PATH / "anomaly_samples.json", "w") as f:
        json.dump(anomaly_data, f, indent=2)


# Initialize test environment
def setup_test_environment():
    """Setup test environment."""
    create_test_data()
    print("Test environment initialized")


# Cleanup test environment
def cleanup_test_environment():
    """Cleanup test environment."""
    import shutil

    if TEST_OUTPUT_PATH.exists():
        shutil.rmtree(TEST_OUTPUT_PATH)
    TEST_OUTPUT_PATH.mkdir(exist_ok=True)

    print("Test environment cleaned up")


if __name__ == "__main__":
    setup_test_environment()
