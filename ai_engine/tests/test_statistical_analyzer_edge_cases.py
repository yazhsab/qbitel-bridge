"""
Edge case tests for Statistical Analyzer.
Focuses on boundary conditions, error handling, and edge scenarios.
"""

import pytest
import pytest_asyncio
import numpy as np

from ai_engine.discovery.statistical_analyzer import (
    StatisticalAnalyzer,
    TrafficPattern,
    FieldBoundary,
    ByteStatistics,
    PatternInfo,
)
from ai_engine.core.config import Config


class TestStatisticalAnalyzerEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return StatisticalAnalyzer(Config())

    @pytest.mark.asyncio
    async def test_empty_traffic_list(self, analyzer):
        """Test with empty traffic list."""
        result = await analyzer.analyze_traffic([])

        assert isinstance(result, TrafficPattern)
        assert result.total_messages == 0
        assert result.message_lengths == []

    @pytest.mark.asyncio
    async def test_single_message(self, analyzer):
        """Test with single message."""
        messages = [b"single_message"]
        result = await analyzer.analyze_traffic(messages)

        assert result.total_messages == 1
        assert len(result.message_lengths) == 1

    @pytest.mark.asyncio
    async def test_empty_messages(self, analyzer):
        """Test with empty messages."""
        messages = [b"", b"", b""]
        result = await analyzer.analyze_traffic(messages)

        assert result.total_messages == 3
        assert all(length == 0 for length in result.message_lengths)

    @pytest.mark.asyncio
    async def test_very_long_message(self, analyzer):
        """Test with very long message."""
        # 10MB message
        long_message = b"A" * (10 * 1024 * 1024)
        result = await analyzer.analyze_traffic([long_message])

        assert result.total_messages == 1
        assert result.message_lengths[0] == len(long_message)

    @pytest.mark.asyncio
    async def test_all_null_bytes(self, analyzer):
        """Test with all null bytes."""
        messages = [b"\x00" * 100] * 10
        result = await analyzer.analyze_traffic(messages)

        assert result.total_messages == 10
        assert result.entropy < 0.5  # Very low entropy

    @pytest.mark.asyncio
    async def test_max_entropy_random_data(self, analyzer):
        """Test with maximum entropy random data."""
        import os

        messages = [os.urandom(1000) for _ in range(10)]
        result = await analyzer.analyze_traffic(messages)

        assert result.entropy > 7.0  # Should be close to 8

    @pytest.mark.asyncio
    async def test_detect_field_boundaries_no_boundaries(self, analyzer):
        """Test field boundary detection with no clear boundaries."""
        messages = [b"sameformateverytime" * 5] * 10
        boundaries = await analyzer.detect_field_boundaries(messages)

        # May find some boundaries or none
        assert isinstance(boundaries, list)

    @pytest.mark.asyncio
    async def test_detect_field_boundaries_single_char_delimiters(self, analyzer):
        """Test with single character delimiters."""
        messages = [b"A|B|C|D", b"E|F|G|H", b"I|J|K|L"]
        boundaries = await analyzer.detect_field_boundaries(messages)

        assert len(boundaries) > 0
        # Should detect | as delimiter

    @pytest.mark.asyncio
    async def test_unicode_in_messages(self, analyzer):
        """Test with Unicode characters (UTF-8 encoded)."""
        messages = [
            "Hello 世界".encode("utf-8"),
            "Test テスト".encode("utf-8"),
            "Data данные".encode("utf-8"),
        ]
        result = await analyzer.analyze_traffic(messages)

        assert result.total_messages == 3

    @pytest.mark.asyncio
    async def test_mixed_binary_and_text(self, analyzer):
        """Test with mixed binary and text data."""
        messages = [
            b"HTTP/1.1 200 OK\r\n",
            b"\x01\x02\x03\x04\x05",
            b"GET /api HTTP/1.1\r\n",
            b"\xff\xfe\xfd\xfc",
        ]
        result = await analyzer.analyze_traffic(messages)

        assert 0 < result.binary_ratio < 1  # Mixed content

    @pytest.mark.asyncio
    async def test_all_identical_messages(self, analyzer):
        """Test with all identical messages."""
        message = b"IDENTICAL_MESSAGE"
        messages = [message] * 100
        result = await analyzer.analyze_traffic(messages)

        assert result.total_messages == 100
        # Low entropy due to repetition

    @pytest.mark.asyncio
    async def test_alternating_patterns(self, analyzer):
        """Test with alternating patterns."""
        messages = [b"PATTERN_A" if i % 2 == 0 else b"PATTERN_B" for i in range(100)]
        result = await analyzer.analyze_traffic(messages)

        assert result.total_messages == 100

    @pytest.mark.asyncio
    async def test_calculate_entropy_edge_cases(self, analyzer):
        """Test entropy calculation edge cases."""
        # Single byte
        entropy1 = await analyzer._calculate_entropy(b"A")
        assert entropy1 == 0.0  # No variety

        # Two different bytes
        entropy2 = await analyzer._calculate_entropy(b"AB")
        assert entropy2 > 0.0

        # Empty data
        entropy_empty = await analyzer._calculate_entropy(b"")
        assert entropy_empty == 0.0

    @pytest.mark.asyncio
    async def test_length_field_detection(self, analyzer):
        """Test detection of length-prefixed messages."""
        messages = [
            b"\x00\x05HelloWorld",  # Length 5, but data is longer
            b"\x00\x04TestData",
            b"\x00\x06Example",
        ]
        boundaries = await analyzer.detect_field_boundaries(messages)

        # Should detect length field pattern
        assert isinstance(boundaries, list)

    @pytest.mark.asyncio
    async def test_nested_delimiters(self, analyzer):
        """Test with nested delimiter structures."""
        messages = [
            b"field1:sub1,sub2;field2:sub3,sub4",
            b"field1:sub5,sub6;field2:sub7,sub8",
        ]
        boundaries = await analyzer.detect_field_boundaries(messages)

        # Should detect multiple delimiter levels
        assert len(boundaries) > 0

    @pytest.mark.asyncio
    async def test_variable_length_fields(self, analyzer):
        """Test with highly variable length fields."""
        messages = [b"A|BB|CCC|DDDD", b"E|FF|GGG|HHHH", b"I|JJ|KKK|LLLL"]
        boundaries = await analyzer.detect_field_boundaries(messages)

        assert len(boundaries) > 0

    @pytest.mark.asyncio
    async def test_messages_with_checksums(self, analyzer):
        """Test messages that include checksums."""
        import struct

        messages = []
        for i in range(10):
            data = f"payload_{i}".encode()
            checksum = sum(data) & 0xFF
            message = data + struct.pack("B", checksum)
            messages.append(message)

        result = await analyzer.analyze_traffic(messages)
        assert result.total_messages == 10

    @pytest.mark.asyncio
    async def test_compressed_data(self, analyzer):
        """Test with compressed/encrypted-looking data."""
        import zlib

        messages = [
            zlib.compress(b"data1"),
            zlib.compress(b"data2"),
            zlib.compress(b"data3"),
        ]
        result = await analyzer.analyze_traffic(messages)

        # Compressed data should have high entropy
        assert result.entropy > 6.0

    @pytest.mark.asyncio
    async def test_messages_with_padding(self, analyzer):
        """Test messages with padding bytes."""
        messages = [
            b"DATA1" + b"\x00" * 10,
            b"DATA2" + b"\x00" * 10,
            b"DATA3" + b"\x00" * 10,
        ]
        boundaries = await analyzer.detect_field_boundaries(messages)

        # Should detect padding pattern
        assert isinstance(boundaries, list)

    @pytest.mark.asyncio
    async def test_high_frequency_rare_bytes(self, analyzer):
        """Test with rare bytes appearing frequently."""
        # Message with many rare control characters
        messages = [b"\x01\x02\x03" + b"normal_text" + b"\x1f\x1e\x1d"] * 10
        result = await analyzer.analyze_traffic(messages)

        assert result.total_messages == 10

    @pytest.mark.asyncio
    async def test_protocol_with_magic_bytes(self, analyzer):
        """Test detection of magic bytes/signatures."""
        messages = [
            b"\x89PNG" + b"rest_of_data",
            b"\x89PNG" + b"different_data",
            b"\x89PNG" + b"more_data",
        ]
        result = await analyzer.analyze_traffic(messages)

        # Should detect common prefix
        assert result.total_messages == 3
        assert len(result.detected_patterns) > 0

    @pytest.mark.asyncio
    async def test_extreme_value_lengths(self, analyzer):
        """Test with extreme variations in message lengths."""
        messages = [
            b"A",  # 1 byte
            b"B" * 10,  # 10 bytes
            b"C" * 1000,  # 1000 bytes
            b"D" * 100000,  # 100KB
        ]
        result = await analyzer.analyze_traffic(messages)

        assert result.total_messages == 4
        assert max(result.message_lengths) == 100000
        assert min(result.message_lengths) == 1


class TestByteStatistics:
    """Test ByteStatistics dataclass."""

    def test_byte_statistics_creation(self):
        """Test creating byte statistics."""
        stats = ByteStatistics(
            frequency={65: 10, 66: 5},
            total_count=15,
            entropy=1.5,
            mean=65.3,
            std_dev=0.5,
            skewness=0.1,
            kurtosis=-0.2,
            chi_square_stat=5.0,
            chi_square_pvalue=0.05,
            is_random=False,
            is_text=True,
            is_binary=False,
        )

        assert stats.total_count == 15
        assert stats.is_text is True
        assert stats.is_binary is False


class TestFieldBoundary:
    """Test FieldBoundary dataclass."""

    def test_field_boundary_creation(self):
        """Test creating field boundary."""
        boundary = FieldBoundary(
            position=5,
            confidence=0.9,
            boundary_type="delimiter",
            evidence={"delimiter": "|"},
            separator=b"|",
        )

        assert boundary.position == 5
        assert boundary.confidence == 0.9
        assert boundary.boundary_type == "delimiter"
        assert boundary.separator == b"|"


class TestPatternInfo:
    """Test PatternInfo dataclass."""

    def test_pattern_info_creation(self):
        """Test creating pattern info."""
        pattern = PatternInfo(
            pattern=b"HTTP",
            frequency=10,
            positions=[0, 100, 200],
            contexts=[b"HTTP/1.1", b"HTTP/2.0"],
            pattern_type="fixed",
            entropy=2.0,
            significance=0.95,
        )

        assert pattern.pattern == b"HTTP"
        assert pattern.frequency == 10
        assert len(pattern.positions) == 3
        assert pattern.pattern_type == "fixed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
