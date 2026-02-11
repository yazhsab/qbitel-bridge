import json
import logging
from types import SimpleNamespace
from datetime import datetime

import pytest

from ai_engine.security.logging import (
    SensitiveDataMasker,
    SecurityLogger,
    StructuredSecurityFormatter,
    SecurityLogType,
    LogLevel,
    get_security_logger,
)
from ai_engine.security.models import SecurityEvent, ThreatAnalysis, ThreatLevel


@pytest.fixture(autouse=True)
def reset_security_logger(monkeypatch):
    import ai_engine.security.logging as module

    module._security_logger = None
    yield
    module._security_logger = None


@pytest.fixture
def stub_config_factory(tmp_path):
    def _factory(*, file_enabled=False, audit_enabled=False, data_masking=True):
        monitoring = SimpleNamespace(
            logging={
                "level": "INFO",
                "console_enabled": False,
                "file_enabled": file_enabled,
                "syslog_enabled": False,
                "log_directory": str(tmp_path),
            }
        )
        security = SimpleNamespace(
            audit={"enabled": audit_enabled},
            privacy={"data_masking": data_masking},
        )
        return SimpleNamespace(monitoring=monitoring, security=security)

    return _factory


def test_sensitive_data_masker_masks_known_patterns():
    masker = SensitiveDataMasker()

    message = "User foo@example.com used password=supersecret from 10.0.0.5"
    masked_message = masker.mask_message(message)

    assert "foo@example.com" not in masked_message
    assert "supersecret" not in masked_message
    assert "10.0.0.5" not in masked_message
    assert "[REDACTED_IP]" in masked_message

    metadata = {
        "Password": "hunter2",
        "nested": {"token": "abc123"},
        "note": "Reach me at bar@example.org",
    }

    masked_metadata = masker.mask_metadata(metadata)
    assert masked_metadata["Password"] == "[REDACTED]"
    assert masked_metadata["nested"]["token"] == "[REDACTED]"
    assert masked_metadata["note"] == "Reach me at [REDACTED]"


def test_security_logger_masks_and_logs_events(monkeypatch, stub_config_factory):
    config = stub_config_factory()
    monkeypatch.setattr(
        "ai_engine.security.logging.get_security_config", lambda: config
    )

    logger = SecurityLogger("test.logger")

    captured_records = []

    def capture(record):
        captured_records.append(record)

    logger.logger.handle = capture  # type: ignore[assignment]

    log_id = logger.log_security_event(
        SecurityLogType.ACCESS_CONTROL,
        "Access granted to foo@example.com with password hunter2",
        metadata={"password": "hunter2", "extra": "Contact foo@example.com"},
    )

    assert log_id
    assert logger.log_counts[LogLevel.INFO.value] == 1
    assert captured_records, "Expected log record to be emitted"

    record = captured_records[0]
    assert "foo@example.com" not in record.message
    assert record.metadata["password"] == "[REDACTED]"
    assert record.metadata["extra"] == "Contact [REDACTED]"

    security_event = SecurityEvent(threat_level=ThreatLevel.CRITICAL)
    logger.log_security_event_obj(security_event)
    assert logger.log_counts[LogLevel.CRITICAL.value] == 1

    threat_analysis = ThreatAnalysis(
        threat_level=ThreatLevel.HIGH, confidence_score=0.9
    )
    logger.log_threat_analysis(threat_analysis)
    assert logger.log_counts[LogLevel.ERROR.value] == 1


def test_security_logger_creates_file_handlers(monkeypatch, stub_config_factory):
    created_handlers = []

    class DummyRotatingFileHandler(logging.Handler):
        def __init__(self, filename, *_, **__):
            super().__init__()
            self.filename = filename
            self.formatter = None
            self._filters = []
            created_handlers.append(self)

        def setFormatter(self, formatter):
            self.formatter = formatter

        def addFilter(self, filter_obj):
            self._filters.append(filter_obj)
            return super().addFilter(filter_obj)

        def emit(self, record):
            return None

    config = stub_config_factory(file_enabled=True, audit_enabled=True)
    monkeypatch.setattr(
        "ai_engine.security.logging.get_security_config", lambda: config
    )
    monkeypatch.setattr(
        logging.handlers, "RotatingFileHandler", DummyRotatingFileHandler
    )

    logger = SecurityLogger("enterprise.logger")

    assert len(created_handlers) == 3  # application, security, audit
    # Security handler should add security filter
    security_handler = created_handlers[1]
    assert any(
        getattr(f, "__name__", "").startswith("_security_log_filter")
        or getattr(f, "__self__", None) is logger
        for f in security_handler._filters
    )
    # Audit handler should include audit filter
    audit_handler = created_handlers[2]
    assert any(
        getattr(f, "__name__", "").startswith("_audit_log_filter")
        or getattr(f, "__self__", None) is logger
        for f in audit_handler._filters
    )


def test_structured_formatter_serializes_record():
    formatter = StructuredSecurityFormatter()

    record = logging.makeLogRecord(
        {
            "name": "qbitel.security",
            "level": logging.INFO,
            "log_id": "123",
            "log_type": SecurityLogType.EVENT_DETECTION.value,
            "event_id": "evt-456",
            "threat_level": ThreatLevel.MEDIUM.value,
            "metadata": {"foo": "bar"},
            "msg": "Sample message",
            "args": (),
            "created": datetime.now().timestamp(),
        }
    )

    payload = json.loads(formatter.format(record))
    assert payload["log_id"] == "123"
    assert payload["metadata"] == {"foo": "bar"}
    assert payload["message"] == "Sample message"
    assert payload["log_type"] == SecurityLogType.EVENT_DETECTION.value


def test_get_security_logger_returns_singleton(monkeypatch, stub_config_factory):
    config = stub_config_factory()
    monkeypatch.setattr(
        "ai_engine.security.logging.get_security_config", lambda: config
    )

    logger_one = get_security_logger("singleton.test")
    logger_two = get_security_logger("singleton.test")

    assert logger_one is logger_two
    assert logger_one.logger_name == "singleton.test"
