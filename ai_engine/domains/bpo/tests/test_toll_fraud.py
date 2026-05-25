"""
Tests for Toll Fraud Detection Module

Tests cover:
- Premium rate number detection
- International call detection
- Off-hours call detection
- Call volume spike detection
- Rapid sequential call detection
- Automated response actions

Note: These tests define the expected behavior for the toll fraud detection
module (ai_engine.domains.bpo.security.toll_fraud). The tests serve as a
specification and will pass once the module is implemented.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from ai_engine.domains.bpo.security.toll_fraud import (
    TollFraudDetector,
    FraudPattern,
    TollFraudRule,
    FraudAction,
    FraudPatternType,
    FraudSeverity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    """Return a default TollFraudDetector."""
    return TollFraudDetector()


@pytest.fixture
def strict_detector():
    """Return a strict TollFraudDetector with tighter thresholds."""
    return TollFraudDetector(
        max_calls_per_minute=5,
        max_international_per_hour=2,
        business_hours_start=8,
        business_hours_end=20,
    )


# ---------------------------------------------------------------------------
# FraudPatternType and FraudSeverity Enums
# ---------------------------------------------------------------------------

class TestFraudEnums:
    """Tests for toll fraud enumeration types."""

    def test_fraud_pattern_types_exist(self):
        """Test fraud pattern types are defined."""
        assert FraudPatternType.PREMIUM_RATE is not None
        assert FraudPatternType.INTERNATIONAL is not None

    def test_fraud_severity_levels(self):
        """Test fraud severity levels are defined."""
        assert FraudSeverity.LOW is not None
        assert FraudSeverity.MEDIUM is not None
        assert FraudSeverity.HIGH is not None
        assert FraudSeverity.CRITICAL is not None

    def test_fraud_action_types(self):
        """Test fraud action types are defined."""
        assert FraudAction.ALERT is not None
        assert FraudAction.BLOCK is not None


# ---------------------------------------------------------------------------
# Premium Rate Number Detection
# ---------------------------------------------------------------------------

class TestPremiumRateDetection:
    """Tests for premium rate number detection."""

    @pytest.mark.parametrize("number", [
        "+19001234567",     # US 900 premium
        "+19761234567",     # US 976 premium
        "+44901234567",     # UK 09 premium
        "+882123456789",    # IRSF 882
        "+883123456789",    # IRSF 883
    ])
    def test_detect_premium_rate_numbers(self, detector, number):
        """Test detection of known premium rate numbers."""
        result = detector.check_number(number)
        assert result.is_premium_rate is True, (
            f"Number {number} should be detected as premium rate"
        )

    @pytest.mark.parametrize("number", [
        "+12125551234",     # US normal
        "+442071234567",    # UK normal
        "+14155551234",     # US normal
    ])
    def test_normal_numbers_not_flagged(self, detector, number):
        """Test normal numbers are not flagged as premium rate."""
        result = detector.check_number(number)
        assert result.is_premium_rate is False, (
            f"Number {number} should not be flagged as premium rate"
        )

    def test_premium_rate_severity(self, detector):
        """Test premium rate calls have HIGH severity."""
        result = detector.check_number("+19001234567")
        assert result.severity in (FraudSeverity.HIGH, FraudSeverity.CRITICAL)

    def test_premium_rate_action_is_block(self, detector):
        """Test premium rate calls trigger BLOCK action."""
        result = detector.check_number("+19001234567")
        assert FraudAction.BLOCK in result.recommended_actions


# ---------------------------------------------------------------------------
# International Call Detection
# ---------------------------------------------------------------------------

class TestInternationalCallDetection:
    """Tests for international call detection."""

    def test_detect_international_call(self, detector):
        """Test detection of international calls."""
        result = detector.check_number("+447911123456")  # UK number
        assert result.is_international is True

    def test_domestic_call_not_international(self, detector):
        """Test domestic calls are not flagged as international."""
        result = detector.check_number("+12125551234")  # US domestic
        assert result.is_international is False

    def test_international_with_restrictions(self, strict_detector):
        """Test international call detection with restrictions applied."""
        result = strict_detector.check_number("+88212345678")  # IRSF
        assert result.is_international is True
        assert result.severity.name in ("HIGH", "CRITICAL")


# ---------------------------------------------------------------------------
# Off-Hours Call Detection
# ---------------------------------------------------------------------------

class TestOffHoursDetection:
    """Tests for off-hours call detection."""

    def test_off_hours_call_flagged(self, strict_detector):
        """Test calls outside business hours are flagged."""
        off_hours_time = datetime.now().replace(hour=3, minute=0)
        result = strict_detector.check_call_time(off_hours_time)
        assert result.is_off_hours is True

    def test_business_hours_call_not_flagged(self, strict_detector):
        """Test calls during business hours are not flagged."""
        business_time = datetime.now().replace(hour=14, minute=0)
        result = strict_detector.check_call_time(business_time)
        assert result.is_off_hours is False

    def test_off_hours_severity(self, strict_detector):
        """Test off-hours calls have appropriate severity."""
        off_hours_time = datetime.now().replace(hour=2, minute=0)
        result = strict_detector.check_call_time(off_hours_time)
        assert result.severity in (FraudSeverity.MEDIUM, FraudSeverity.HIGH)


# ---------------------------------------------------------------------------
# Call Volume Spike Detection
# ---------------------------------------------------------------------------

class TestVolumeSpike:
    """Tests for call volume spike detection."""

    def test_detect_volume_spike(self, detector):
        """Test detection of unusual call volume spikes."""
        # Simulate a spike in call volume
        call_events = []
        base_time = datetime.now()
        for i in range(100):
            call_events.append({
                "call_id": f"CALL-{i}",
                "timestamp": base_time + timedelta(seconds=i),
                "source": "AGENT-001",
                "destination": f"+1212555{i:04d}",
            })
        result = detector.check_volume(call_events, window_minutes=1)
        assert result.is_spike is True
        assert result.call_count > 50

    def test_normal_volume_not_flagged(self, detector):
        """Test normal call volume is not flagged."""
        call_events = []
        base_time = datetime.now()
        for i in range(5):
            call_events.append({
                "call_id": f"CALL-{i}",
                "timestamp": base_time + timedelta(minutes=i * 5),
                "source": "AGENT-001",
                "destination": f"+1212555{i:04d}",
            })
        result = detector.check_volume(call_events, window_minutes=60)
        assert result.is_spike is False


# ---------------------------------------------------------------------------
# Rapid Sequential Call Detection
# ---------------------------------------------------------------------------

class TestRapidSequentialCalls:
    """Tests for rapid sequential call detection."""

    def test_detect_rapid_calls(self, detector):
        """Test detection of rapid sequential calls from same source."""
        call_events = []
        base_time = datetime.now()
        for i in range(20):
            call_events.append({
                "call_id": f"CALL-{i}",
                "timestamp": base_time + timedelta(seconds=i * 2),
                "source": "AGENT-SUSPECT",
                "destination": f"+1900555{i:04d}",
                "duration_seconds": 5,
            })
        result = detector.check_rapid_calls(
            call_events,
            source="AGENT-SUSPECT",
            window_seconds=60,
        )
        assert result.is_suspicious is True
        assert result.pattern_type == FraudPatternType.RAPID_SEQUENTIAL

    def test_normal_call_pace_not_flagged(self, detector):
        """Test normal call pace is not flagged."""
        call_events = []
        base_time = datetime.now()
        for i in range(3):
            call_events.append({
                "call_id": f"CALL-{i}",
                "timestamp": base_time + timedelta(minutes=i * 10),
                "source": "AGENT-NORMAL",
                "destination": f"+1212555{i:04d}",
                "duration_seconds": 300,
            })
        result = detector.check_rapid_calls(
            call_events,
            source="AGENT-NORMAL",
            window_seconds=3600,
        )
        assert result.is_suspicious is False


# ---------------------------------------------------------------------------
# Automated Response Actions
# ---------------------------------------------------------------------------

class TestAutomatedResponses:
    """Tests for automated fraud response actions."""

    def test_block_action_on_premium_rate(self, detector):
        """Test BLOCK action is recommended for premium rate calls."""
        result = detector.check_number("+19001234567")
        assert FraudAction.BLOCK in result.recommended_actions

    def test_alert_action_on_suspicious(self, detector):
        """Test ALERT action is included for suspicious activity."""
        off_hours_time = datetime.now().replace(hour=3, minute=0)
        result = detector.check_call_time(off_hours_time)
        assert FraudAction.ALERT in result.recommended_actions

    def test_fraud_pattern_contains_details(self, detector):
        """Test fraud pattern result contains detection details."""
        result = detector.check_number("+19001234567")
        assert isinstance(result, FraudPattern)
        assert result.description is not None
        assert len(result.description) > 0

    def test_toll_fraud_rule_creation(self):
        """Test creating a custom toll fraud rule."""
        rule = TollFraudRule(
            name="Block IRSF",
            pattern_type=FraudPatternType.PREMIUM_RATE,
            severity=FraudSeverity.CRITICAL,
            action=FraudAction.BLOCK,
            description="Block International Revenue Share Fraud numbers",
        )
        assert rule.name == "Block IRSF"
        assert rule.severity == FraudSeverity.CRITICAL
        assert rule.action == FraudAction.BLOCK


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestTollFraudIntegration:
    """Integration tests for toll fraud detection."""

    def test_full_call_screening(self, detector):
        """Test full call screening workflow."""
        # Screen an outbound call
        number_check = detector.check_number("+19001234567")
        assert number_check.is_premium_rate is True

        # Verify the detection includes actionable information
        assert number_check.severity is not None
        assert len(number_check.recommended_actions) > 0

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert isinstance(detector, TollFraudDetector)
