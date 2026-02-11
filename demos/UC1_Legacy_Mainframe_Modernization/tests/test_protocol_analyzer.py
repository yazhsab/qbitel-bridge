"""
Unit tests for protocol analysis functionality.
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


class TestBasicProtocolAnalyzer:
    """Tests for basic protocol analysis (fallback mode)."""

    def test_detect_fixed_length_messages(self):
        """Test detection of fixed-length messages."""
        from production_app import _basic_protocol_analysis

        # All messages same length (16 bytes)
        samples = [
            bytes.fromhex("01020304" + "00" * 12),
            bytes.fromhex("01020304" + "11" * 12),
            bytes.fromhex("01020304" + "22" * 12),
        ]

        result = _basic_protocol_analysis(samples)
        patterns = result["patterns"]

        # Should detect fixed_length pattern
        pattern_types = [p.get("pattern_type") for p in patterns]
        assert "fixed_length" in pattern_types

    def test_detect_magic_number(self):
        """Test detection of magic number."""
        from production_app import _basic_protocol_analysis

        # All messages start with same 4 bytes
        samples = [
            bytes.fromhex("deadbeef" + "0001" + "48656c6c6f"),
            bytes.fromhex("deadbeef" + "0002" + "576f726c64"),
            bytes.fromhex("deadbeef" + "0003" + "546573743132"),
        ]

        result = _basic_protocol_analysis(samples)
        patterns = result["patterns"]

        # Should detect magic_number pattern
        pattern_types = [p.get("pattern_type") for p in patterns]
        assert "magic_number" in pattern_types

        # Find the magic number pattern
        magic_patterns = [p for p in patterns if p.get("pattern_type") == "magic_number"]
        assert len(magic_patterns) > 0
        assert "deadbeef" in magic_patterns[0]["description"]

    def test_detect_binary_protocol(self):
        """Test detection of binary protocol."""
        from production_app import _basic_protocol_analysis

        # Binary data (non-ASCII)
        samples = [
            bytes([0x00, 0x01, 0x02, 0x80, 0x90, 0xFF]),
            bytes([0x00, 0x01, 0x02, 0x81, 0x91, 0xFE]),
            bytes([0x00, 0x01, 0x02, 0x82, 0x92, 0xFD]),
        ]

        result = _basic_protocol_analysis(samples)
        assert result["characteristics"]["is_binary"] is True

    def test_detect_text_protocol(self):
        """Test detection of text-based protocol."""
        from production_app import _basic_protocol_analysis

        # ASCII text data
        samples = [
            b"GET /api/users HTTP/1.1",
            b"POST /api/data HTTP/1.1",
            b"PUT /api/item HTTP/1.1",
        ]

        result = _basic_protocol_analysis(samples)
        assert result["characteristics"]["is_binary"] is False

    def test_generate_field_definitions(self):
        """Test generation of field definitions."""
        from production_app import _basic_protocol_analysis

        samples = [
            bytes.fromhex("01020304000548656c6c6f"),  # header + len + "Hello"
        ]

        result = _basic_protocol_analysis(samples)
        fields = result["fields"]

        assert len(fields) >= 1
        # Should have at least header field
        field_names = [f["name"] for f in fields]
        assert "header" in field_names

    def test_confidence_score_range(self):
        """Test confidence score is in valid range."""
        from production_app import _basic_protocol_analysis

        samples = [
            bytes.fromhex("01020304" + "00" * 10),
            bytes.fromhex("01020304" + "11" * 10),
        ]

        result = _basic_protocol_analysis(samples)
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_empty_samples(self):
        """Test handling of empty samples list."""
        from production_app import _basic_protocol_analysis

        result = _basic_protocol_analysis([])
        assert "protocol_name" in result
        assert "characteristics" in result

    def test_single_sample(self):
        """Test handling of single sample."""
        from production_app import _basic_protocol_analysis

        samples = [bytes.fromhex("01020304050607080910")]

        result = _basic_protocol_analysis(samples)
        assert "protocol_name" in result
        # Confidence should be lower with single sample
        assert result["confidence_score"] < 1.0

    def test_message_types_identified(self):
        """Test identification of message types."""
        from production_app import _basic_protocol_analysis

        samples = [
            bytes.fromhex("01020304" + "00" * 10),
            bytes.fromhex("01020304" + "11" * 10),
        ]

        result = _basic_protocol_analysis(samples)
        assert "message_types" in result
        assert len(result["message_types"]) >= 1

    def test_characteristics_structure(self):
        """Test characteristics structure."""
        from production_app import _basic_protocol_analysis

        samples = [bytes.fromhex("01020304050607080910")]

        result = _basic_protocol_analysis(samples)
        chars = result["characteristics"]

        assert "is_binary" in chars
        assert "is_stateful" in chars
        assert "uses_encryption" in chars
        assert "has_checksums" in chars

        # All should be booleans
        assert isinstance(chars["is_binary"], bool)
        assert isinstance(chars["is_stateful"], bool)
        assert isinstance(chars["uses_encryption"], bool)
        assert isinstance(chars["has_checksums"], bool)


class TestProtocolPatternRecognition:
    """Tests for protocol pattern recognition."""

    def test_recognize_length_prefixed_messages(self):
        """Test recognition of length-prefixed messages."""
        from production_app import _basic_protocol_analysis

        # Messages with length field (big-endian 16-bit)
        samples = [
            bytes.fromhex("0005") + b"Hello",  # Length 5
            bytes.fromhex("0006") + b"World!",  # Length 6
            bytes.fromhex("0004") + b"Test",  # Length 4
        ]

        result = _basic_protocol_analysis(samples)
        # Should identify length field
        fields = result["fields"]
        field_types = [f.get("field_type") for f in fields]
        assert "integer" in field_types or "binary" in field_types

    def test_variable_length_messages(self):
        """Test handling of variable-length messages."""
        from production_app import _basic_protocol_analysis

        samples = [
            bytes.fromhex("01" * 10),
            bytes.fromhex("02" * 20),
            bytes.fromhex("03" * 15),
        ]

        result = _basic_protocol_analysis(samples)
        # Should not detect fixed_length pattern
        pattern_types = [p.get("pattern_type") for p in result["patterns"]]
        assert "fixed_length" not in pattern_types


class TestProtocolDocumentation:
    """Tests for protocol documentation generation."""

    def test_documentation_generated(self):
        """Test that documentation is generated."""
        from production_app import _basic_protocol_analysis

        samples = [
            bytes.fromhex("01020304050607080910"),
        ]

        result = _basic_protocol_analysis(samples)
        assert "documentation" in result
        assert len(result["documentation"]) > 0

    def test_complexity_assessment(self):
        """Test complexity assessment."""
        from production_app import _basic_protocol_analysis

        samples = [
            bytes.fromhex("01020304050607080910"),
        ]

        result = _basic_protocol_analysis(samples)
        assert "complexity" in result
        assert result["complexity"] in ["simple", "moderate", "complex", "highly_complex"]
