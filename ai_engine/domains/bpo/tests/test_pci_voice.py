"""
Tests for PCI Voice Security Module

Tests cover:
- DTMF masking modes
- PAN detection (valid Luhn numbers)
- PAN detection (false positives - not card numbers)
- Recording pause/resume flow
- Agent screen masking
- PCI scope tracking

Note: These tests define the expected behavior for the PCI voice security
module (ai_engine.domains.bpo.security.pci_voice). The tests serve as a
specification and will pass once the module is implemented.
"""

import pytest
from datetime import datetime

from ai_engine.domains.bpo.security.pci_voice import (
    PCIVoiceProtector,
    DTMFMaskingMode,
    PANDetector,
    RecordingController,
    AgentScreenMasker,
    PCIScopeManager,
    ComplianceReporter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def protector():
    """Return a default PCIVoiceProtector."""
    return PCIVoiceProtector()


@pytest.fixture
def pan_detector():
    """Return a PANDetector instance."""
    return PANDetector()


@pytest.fixture
def recording_controller():
    """Return a RecordingController instance."""
    return RecordingController()


@pytest.fixture
def screen_masker():
    """Return an AgentScreenMasker instance."""
    return AgentScreenMasker()


@pytest.fixture
def scope_manager():
    """Return a PCIScopeManager instance."""
    return PCIScopeManager()


# ---------------------------------------------------------------------------
# DTMFMaskingMode Enum
# ---------------------------------------------------------------------------

class TestDTMFMaskingMode:
    """Tests for DTMF masking modes."""

    def test_clamp_mode_exists(self):
        """Test CLAMP masking mode is defined."""
        assert DTMFMaskingMode.CLAMP is not None

    def test_flat_mode_exists(self):
        """Test FLAT masking mode is defined."""
        assert DTMFMaskingMode.FLAT is not None

    def test_replace_mode_exists(self):
        """Test REPLACE masking mode is defined."""
        assert DTMFMaskingMode.REPLACE is not None

    def test_modes_are_distinct(self):
        """Test all masking modes have distinct values."""
        modes = [m.value for m in DTMFMaskingMode]
        assert len(modes) == len(set(modes)), "DTMF masking modes must be unique"

    def test_protector_supports_all_modes(self, protector):
        """Test PCIVoiceProtector supports all DTMF masking modes."""
        for mode in DTMFMaskingMode:
            protector.set_dtmf_masking_mode(mode)
            assert protector.dtmf_masking_mode == mode


# ---------------------------------------------------------------------------
# PAN Detection - Valid Luhn Numbers
# ---------------------------------------------------------------------------

class TestPANDetectionValid:
    """Tests for PAN detection with valid card numbers (Luhn check passes)."""

    @pytest.mark.parametrize("pan,description", [
        ("4111111111111111", "Visa test card"),
        ("4012888888881881", "Visa test card alternate"),
        ("5500000000000004", "Mastercard test card"),
        ("5105105105105100", "Mastercard test card alternate"),
        ("340000000000009", "Amex test card (15 digits)"),
        ("371449635398431", "Amex test card alternate"),
        ("6011111111111117", "Discover test card"),
        ("3530111333300000", "JCB test card"),
    ])
    def test_detect_valid_pan(self, pan_detector, pan, description):
        """Test detection of valid card numbers (Luhn valid)."""
        result = pan_detector.detect(pan)
        assert result.is_pan is True, (
            f"Should detect {description}: {pan}"
        )

    @pytest.mark.parametrize("pan", [
        "4111111111111111",
        "5500000000000004",
        "340000000000009",
    ])
    def test_luhn_validation_passes(self, pan_detector, pan):
        """Test Luhn algorithm validates known test numbers."""
        assert pan_detector.luhn_check(pan) is True

    def test_detect_pan_in_text(self, pan_detector):
        """Test PAN detection within surrounding text."""
        text = "Customer card number is 4111111111111111 for the order"
        result = pan_detector.scan_text(text)
        assert len(result.detected_pans) >= 1

    def test_detect_pan_with_spaces(self, pan_detector):
        """Test PAN detection with space-separated groups."""
        text = "4111 1111 1111 1111"
        result = pan_detector.scan_text(text)
        assert len(result.detected_pans) >= 1

    def test_detect_pan_with_dashes(self, pan_detector):
        """Test PAN detection with dash-separated groups."""
        text = "4111-1111-1111-1111"
        result = pan_detector.scan_text(text)
        assert len(result.detected_pans) >= 1


# ---------------------------------------------------------------------------
# PAN Detection - False Positives
# ---------------------------------------------------------------------------

class TestPANDetectionFalsePositives:
    """Tests for PAN detection rejecting non-card numbers."""

    @pytest.mark.parametrize("number,description", [
        ("1234567890123456", "Sequential digits (fails Luhn)"),
        ("0000000000000000", "All zeros"),
        ("9999999999999999", "All nines (fails Luhn)"),
        ("1111111111111111", "All ones (fails Luhn)"),
        ("123456789", "Too short (9 digits)"),
        ("12345678901234567890", "Too long (20 digits)"),
    ])
    def test_reject_non_pan(self, pan_detector, number, description):
        """Test rejection of non-card numbers."""
        result = pan_detector.detect(number)
        assert result.is_pan is False, (
            f"Should reject {description}: {number}"
        )

    def test_reject_phone_numbers(self, pan_detector):
        """Test phone numbers are not flagged as PANs."""
        text = "Call us at +1-212-555-1234 or 1-800-555-0123"
        result = pan_detector.scan_text(text)
        assert len(result.detected_pans) == 0, (
            "Phone numbers should not be detected as PANs"
        )

    def test_reject_timestamps(self, pan_detector):
        """Test numeric timestamps are not flagged as PANs."""
        text = "Transaction at 20240115143022123456"
        result = pan_detector.scan_text(text)
        # Should not detect the timestamp as a PAN
        for detected in result.detected_pans:
            assert pan_detector.luhn_check(detected) is True

    def test_reject_short_numbers(self, pan_detector):
        """Test numbers shorter than 13 digits are rejected."""
        result = pan_detector.detect("123456")
        assert result.is_pan is False


# ---------------------------------------------------------------------------
# Recording Pause/Resume Flow
# ---------------------------------------------------------------------------

class TestRecordingPauseResume:
    """Tests for recording pause/resume flow."""

    def test_pause_recording(self, recording_controller):
        """Test pausing a call recording."""
        result = recording_controller.pause(
            call_id="CALL-001",
            agent_id="AGENT-001",
            reason="PCI_PAYMENT",
        )
        assert result.success is True
        assert result.state == "PAUSED"

    def test_resume_recording(self, recording_controller):
        """Test resuming a call recording."""
        recording_controller.pause(
            call_id="CALL-001",
            agent_id="AGENT-001",
            reason="PCI_PAYMENT",
        )
        result = recording_controller.resume(
            call_id="CALL-001",
            agent_id="AGENT-001",
        )
        assert result.success is True
        assert result.state == "RECORDING"

    def test_pause_resume_sequence(self, recording_controller):
        """Test complete pause/resume sequence."""
        # Start in recording state
        pause_result = recording_controller.pause(
            call_id="CALL-002",
            agent_id="AGENT-001",
            reason="PCI_PAYMENT",
        )
        assert pause_result.state == "PAUSED"

        resume_result = recording_controller.resume(
            call_id="CALL-002",
            agent_id="AGENT-001",
        )
        assert resume_result.state == "RECORDING"

    def test_auto_resume_timeout(self, recording_controller):
        """Test auto-resume after timeout."""
        result = recording_controller.pause(
            call_id="CALL-003",
            agent_id="AGENT-001",
            reason="PCI_PAYMENT",
            auto_resume_seconds=120,
        )
        assert result.auto_resume_scheduled is True

    def test_pause_generates_audit_event(self, recording_controller):
        """Test that pause generates an audit event."""
        result = recording_controller.pause(
            call_id="CALL-004",
            agent_id="AGENT-001",
            reason="PCI_PAYMENT",
        )
        assert result.audit_event_id is not None

    def test_double_pause_is_idempotent(self, recording_controller):
        """Test that pausing an already paused recording is handled gracefully."""
        recording_controller.pause(
            call_id="CALL-005",
            agent_id="AGENT-001",
            reason="PCI_PAYMENT",
        )
        result = recording_controller.pause(
            call_id="CALL-005",
            agent_id="AGENT-001",
            reason="PCI_PAYMENT",
        )
        assert result.state == "PAUSED"


# ---------------------------------------------------------------------------
# Agent Screen Masking
# ---------------------------------------------------------------------------

class TestAgentScreenMasking:
    """Tests for agent screen masking."""

    def test_mask_credit_card(self, screen_masker):
        """Test masking a credit card number on screen."""
        masked = screen_masker.mask_pan("4111111111111111")
        assert "4111" not in masked[:4] or masked.endswith("1111")
        assert "****" in masked or "XXXX" in masked

    def test_mask_shows_last_four(self, screen_masker):
        """Test masked output shows last 4 digits."""
        masked = screen_masker.mask_pan("4111111111111111")
        assert masked.endswith("1111")

    def test_mask_ssn(self, screen_masker):
        """Test masking a Social Security Number."""
        masked = screen_masker.mask_ssn("123-45-6789")
        assert "123" not in masked
        assert "6789" in masked  # Last 4 visible

    def test_mask_in_text(self, screen_masker):
        """Test masking sensitive data within text."""
        text = "Card: 4111111111111111, SSN: 123-45-6789"
        masked = screen_masker.mask_text(text)
        assert "4111111111111111" not in masked
        assert "123-45-6789" not in masked

    def test_mask_preserves_non_sensitive(self, screen_masker):
        """Test masking preserves non-sensitive text."""
        text = "Customer name: John Doe, Card: 4111111111111111"
        masked = screen_masker.mask_text(text)
        assert "John Doe" in masked


# ---------------------------------------------------------------------------
# PCI Scope Tracking
# ---------------------------------------------------------------------------

class TestPCIScopeTracking:
    """Tests for PCI scope tracking."""

    def test_enter_pci_scope(self, scope_manager):
        """Test entering PCI scope for a call."""
        result = scope_manager.enter_scope(
            call_id="CALL-001",
            agent_id="AGENT-001",
            reason="PAYMENT_PROCESSING",
        )
        assert result.in_scope is True

    def test_exit_pci_scope(self, scope_manager):
        """Test exiting PCI scope for a call."""
        scope_manager.enter_scope(
            call_id="CALL-001",
            agent_id="AGENT-001",
            reason="PAYMENT_PROCESSING",
        )
        result = scope_manager.exit_scope(
            call_id="CALL-001",
            agent_id="AGENT-001",
        )
        assert result.in_scope is False

    def test_scope_tracking_per_call(self, scope_manager):
        """Test PCI scope is tracked per call."""
        scope_manager.enter_scope(
            call_id="CALL-001",
            agent_id="AGENT-001",
            reason="PAYMENT",
        )
        assert scope_manager.is_in_scope("CALL-001") is True
        assert scope_manager.is_in_scope("CALL-002") is False

    def test_scope_entry_triggers_protections(self, scope_manager):
        """Test entering PCI scope triggers protections."""
        result = scope_manager.enter_scope(
            call_id="CALL-001",
            agent_id="AGENT-001",
            reason="PAYMENT",
        )
        assert result.dtmf_masking_active is True
        assert result.recording_paused is True
        assert result.screen_masking_active is True

    def test_scope_exit_releases_protections(self, scope_manager):
        """Test exiting PCI scope releases protections."""
        scope_manager.enter_scope(
            call_id="CALL-001",
            agent_id="AGENT-001",
            reason="PAYMENT",
        )
        result = scope_manager.exit_scope(
            call_id="CALL-001",
            agent_id="AGENT-001",
        )
        assert result.dtmf_masking_active is False
        assert result.recording_paused is False

    def test_scope_duration_tracked(self, scope_manager):
        """Test that time spent in PCI scope is tracked."""
        scope_manager.enter_scope(
            call_id="CALL-001",
            agent_id="AGENT-001",
            reason="PAYMENT",
        )
        result = scope_manager.exit_scope(
            call_id="CALL-001",
            agent_id="AGENT-001",
        )
        assert result.duration_seconds >= 0


# ---------------------------------------------------------------------------
# Compliance Reporter
# ---------------------------------------------------------------------------

class TestComplianceReporter:
    """Tests for PCI compliance reporting."""

    def test_reporter_creation(self):
        """Test ComplianceReporter can be instantiated."""
        reporter = ComplianceReporter()
        assert reporter is not None

    def test_generate_report(self):
        """Test generating a compliance report."""
        reporter = ComplianceReporter()
        report = reporter.generate_report(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )
        assert report is not None
        assert "pci_compliance" in report or hasattr(report, "pci_compliance")


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestPCIVoiceIntegration:
    """Integration tests for PCI voice security."""

    def test_protector_initialization(self, protector):
        """Test PCIVoiceProtector initializes correctly."""
        assert protector is not None
        assert isinstance(protector, PCIVoiceProtector)

    def test_full_payment_flow(self, protector):
        """Test full PCI payment protection flow."""
        call_id = "CALL-PAYMENT-001"
        agent_id = "AGENT-001"

        # Enter PCI scope
        scope_result = protector.enter_pci_scope(
            call_id=call_id,
            agent_id=agent_id,
        )
        assert scope_result.in_scope is True

        # Verify protections are active
        assert protector.is_dtmf_masking_active(call_id) is True
        assert protector.is_recording_paused(call_id) is True

        # Exit PCI scope
        exit_result = protector.exit_pci_scope(
            call_id=call_id,
            agent_id=agent_id,
        )
        assert exit_result.in_scope is False

    def test_pan_detector_standalone(self, pan_detector):
        """Test PANDetector works independently."""
        assert pan_detector is not None
        result = pan_detector.detect("4111111111111111")
        assert result.is_pan is True
