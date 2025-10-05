"""
CRONOS AI Engine - Statistical Analyzer for Protocol Discovery

This module implements enterprise-grade statistical analysis for protocol discovery,
including frequency analysis, entropy calculation, pattern detection, and field boundary identification.
"""

import asyncio
import logging
import string
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import stats
from scipy.stats import entropy
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import hashlib
import struct

from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException


@dataclass
class ByteStatistics:
    """Statistical information about byte distribution."""

    frequency: Dict[int, int]
    total_count: int
    entropy: float
    mean: float
    std_dev: float
    skewness: float
    kurtosis: float
    chi_square_stat: float
    chi_square_pvalue: float
    is_random: bool
    is_text: bool
    is_binary: bool


@dataclass
class ByteFrequency(ByteStatistics):
    """Backward-compatible alias of ``ByteStatistics`` for legacy imports."""


@dataclass
class FieldStatistics:
    """Aggregated statistics for a logical protocol field."""

    min_length: int
    max_length: int
    avg_length: float
    length_std_dev: float
    value_frequency: Dict[str, int]
    unique_values: int
    entropy: float
    is_fixed_length: bool
    is_printable: bool
    is_numeric: bool


@dataclass
class FieldBoundary:
    """Represents a potential field boundary."""

    position: int
    confidence: float
    boundary_type: str  # 'delimiter', 'length_field', 'entropy_change', 'pattern_break'
    evidence: Dict[str, Any]
    separator: Optional[bytes] = None


@dataclass
class PatternInfo:
    """Information about a detected pattern."""

    pattern: bytes
    frequency: int
    positions: List[int]
    contexts: List[bytes]
    pattern_type: str  # 'fixed', 'repeated', 'alternating', 'structured'
    entropy: float
    significance: float


@dataclass
class StructuralFeatures:
    """Structural features of message(s)."""

    length_distribution: Dict[int, int]
    common_prefixes: List[Tuple[bytes, int]]
    common_suffixes: List[Tuple[bytes, int]]
    repeating_sequences: List[PatternInfo]
    field_boundaries: List[FieldBoundary]
    header_length: Optional[int]
    footer_length: Optional[int]
    is_fixed_length: bool
    is_variable_length: bool
    message_classes: List[str]


@dataclass
class TrafficPattern:
    """Aggregate metrics describing analyzed traffic."""

    total_messages: int
    message_lengths: List[int]
    entropy: float
    binary_ratio: float
    detected_patterns: List[PatternInfo]
    field_boundaries: List[FieldBoundary]
    processing_time: float


class StatisticalAnalyzer:
    """
    Enterprise-grade statistical analyzer for protocol discovery.

    This class implements comprehensive statistical analysis techniques for
    automated protocol structure discovery and field boundary detection.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the statistical analyzer."""
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)

        # Analysis parameters
        self.min_pattern_length = 2
        self.max_pattern_length = 32
        self.min_pattern_frequency = 3
        self.entropy_window_size = 8
        self.boundary_confidence_threshold = 0.7
        self.clustering_eps = 0.3
        self.clustering_min_samples = 3

        # Performance optimization
        self.use_parallel_processing = True
        self.max_workers = (
            config.inference.num_workers if hasattr(config, "inference") else 4
        )
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Cache for performance
        self._pattern_cache: Dict[str, PatternInfo] = {}
        self._boundary_cache: Dict[str, List[FieldBoundary]] = {}

        self.logger.info("Statistical Analyzer initialized")

    async def analyze_traffic(self, messages: List[bytes]) -> TrafficPattern:
        """Analyze message traffic and return high level metrics."""
        if not messages:
            raise ProtocolException("No messages provided for analysis")

        start_time = time.time()

        byte_stats_task = asyncio.create_task(self._compute_byte_statistics(b"".join(messages)))
        pattern_task = asyncio.create_task(self.detect_patterns(messages))
        boundary_task = asyncio.create_task(self.detect_field_boundaries(messages))

        byte_stats, patterns, boundaries = await asyncio.gather(
            byte_stats_task, pattern_task, boundary_task
        )

        processing_time = time.time() - start_time
        message_lengths = [len(msg) for msg in messages]
        entropy_value = await self._calculate_entropy(b"".join(messages))
        binary_ratio = self._calculate_binary_ratio(messages)

        traffic_pattern = TrafficPattern(
            total_messages=len(messages),
            message_lengths=message_lengths,
            entropy=entropy_value,
            binary_ratio=binary_ratio,
            detected_patterns=patterns,
            field_boundaries=boundaries,
            processing_time=processing_time,
        )

        self.logger.debug(
            "Traffic analysis completed",
            extra={
                "total_messages": traffic_pattern.total_messages,
                "entropy": traffic_pattern.entropy,
                "binary_ratio": traffic_pattern.binary_ratio,
                "processing_time": traffic_pattern.processing_time,
            },
        )
        return traffic_pattern

    async def analyze_messages(self, messages: List[bytes]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on message samples.

        Args:
            messages: List of protocol message samples

        Returns:
            Comprehensive analysis results including statistics and structure
        """
        if not messages:
            raise ProtocolException("Empty message list provided")

        start_time = time.time()
        self.logger.info(f"Starting statistical analysis on {len(messages)} messages")

        try:
            # Parallel analysis tasks
            tasks = []

            # Byte-level statistics
            tasks.append(self._analyze_byte_statistics(messages))

            # Structural analysis
            tasks.append(self._analyze_structure(messages))

            # Pattern detection
            tasks.append(self._detect_patterns(messages))

            # Field boundary detection
            tasks.append(self._detect_field_boundaries(messages))

            # Message classification
            tasks.append(self._classify_messages(messages))

            # Execute all analyses in parallel
            results = await asyncio.gather(*tasks)

            byte_stats, structural_features, patterns, boundaries, classifications = (
                results
            )

            # Combine results
            analysis_result = {
                "byte_statistics": byte_stats,
                "structural_features": structural_features,
                "patterns": patterns,
                "field_boundaries": boundaries,
                "message_classifications": classifications,
                "metadata": {
                    "num_messages": len(messages),
                    "total_bytes": sum(len(msg) for msg in messages),
                    "analysis_time": time.time() - start_time,
                    "analyzer_version": "1.0.0",
                },
            }

            self.logger.info(
                f"Statistical analysis completed in {time.time() - start_time:.2f}s"
            )
            return analysis_result

        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            raise ModelException(f"Statistical analysis error: {e}")

    async def _analyze_byte_statistics(
        self, messages: List[bytes]
    ) -> Dict[str, ByteStatistics]:
        """Analyze byte-level statistics for all messages."""
        self.logger.debug("Analyzing byte statistics")

        stats_per_position = {}
        all_bytes = b"".join(messages)

        # Overall statistics
        overall_stats = await self._compute_byte_statistics(all_bytes)
        stats_per_position["overall"] = overall_stats

        # Position-based statistics (for fixed-length protocols)
        if messages and self._is_likely_fixed_length(messages):
            msg_length = len(messages[0])
            for pos in range(msg_length):
                position_bytes = bytes([msg[pos] for msg in messages if pos < len(msg)])
                if position_bytes:
                    pos_stats = await self._compute_byte_statistics(position_bytes)
                    stats_per_position[f"position_{pos}"] = pos_stats

        return stats_per_position

    async def _compute_byte_statistics(self, data: bytes) -> ByteStatistics:
        """Compute comprehensive byte statistics."""
        if not data:
            return ByteStatistics(
                frequency={},
                total_count=0,
                entropy=0.0,
                mean=0.0,
                std_dev=0.0,
                skewness=0.0,
                kurtosis=0.0,
                chi_square_stat=0.0,
                chi_square_pvalue=1.0,
                is_random=False,
                is_text=False,
                is_binary=True,
            )

        # Frequency analysis
        frequency = Counter(data)
        total_count = len(data)

        # Convert to probabilities for entropy calculation
        probabilities = np.array([freq / total_count for freq in frequency.values()])
        byte_entropy = entropy(probabilities, base=2)

        # Basic statistics
        data_array = np.array(list(data), dtype=np.float64)
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        skew_val = stats.skew(data_array)
        kurt_val = stats.kurtosis(data_array)

        # Chi-square test for randomness (uniform distribution)
        expected_freq = total_count / 256
        observed_freq = np.array([frequency.get(i, 0) for i in range(256)])
        chi2_stat, chi2_p = stats.chisquare(observed_freq, f_exp=expected_freq)

        # Content classification
        is_random = byte_entropy > 7.5  # High entropy suggests randomness/encryption
        is_text = self._is_likely_text(data)
        is_binary = not is_text and byte_entropy < 7.0

        return ByteStatistics(
            frequency=dict(frequency),
            total_count=total_count,
            entropy=byte_entropy,
            mean=mean_val,
            std_dev=std_val,
            skewness=skew_val,
            kurtosis=kurt_val,
            chi_square_stat=chi2_stat,
            chi_square_pvalue=chi2_p,
            is_random=is_random,
            is_text=is_text,
            is_binary=is_binary,
        )

    async def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy for provided data."""
        if not data:
            return 0.0

        frequency = Counter(data)
        total = len(data)
        probabilities = np.array([count / total for count in frequency.values()])
        return float(entropy(probabilities, base=2))

    def _calculate_binary_ratio(self, messages: List[bytes]) -> float:
        """Estimate ratio of binary messages in sample."""
        if not messages:
            return 0.0

        binary_count = 0
        for msg in messages:
            if not msg:
                continue
            printable = sum(1 for b in msg if 32 <= b <= 126)
            if printable / max(len(msg), 1) < 0.6:
                binary_count += 1

        return binary_count / len(messages)

    def summarize_field_statistics(self, field_values: List[bytes]) -> FieldStatistics:
        """Summarize discrete protocol field samples into statistical features."""
        if not field_values:
            return FieldStatistics(
                min_length=0,
                max_length=0,
                avg_length=0.0,
                length_std_dev=0.0,
                value_frequency={},
                unique_values=0,
                entropy=0.0,
                is_fixed_length=True,
                is_printable=False,
                is_numeric=False,
            )

        lengths = np.array([len(value) for value in field_values], dtype=np.float64)
        min_len = int(lengths.min())
        max_len = int(lengths.max())
        avg_len = float(lengths.mean())
        length_std_dev = float(lengths.std())

        normalized_values = [self._normalize_field_value(value) for value in field_values]
        frequency = Counter(normalized_values)
        total = sum(frequency.values())

        distribution = (
            np.array(list(frequency.values()), dtype=np.float64) / total
            if total
            else np.array([])
        )
        entropy_value = float(entropy(distribution, base=2)) if distribution.size else 0.0

        is_fixed_length = min_len == max_len
        is_printable = all(
            self._is_printable(value) for value in field_values if len(value) > 0
        )
        is_numeric = all(
            self._is_numeric(value) for value in field_values if len(value) > 0
        )

        return FieldStatistics(
            min_length=min_len,
            max_length=max_len,
            avg_length=avg_len,
            length_std_dev=length_std_dev,
            value_frequency=dict(frequency),
            unique_values=len(frequency),
            entropy=entropy_value,
            is_fixed_length=is_fixed_length,
            is_printable=is_printable,
            is_numeric=is_numeric,
        )

    def _normalize_field_value(self, value: bytes) -> str:
        """Normalize raw field bytes into a readable key for counting."""
        if not value:
            return ""

        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError:
            decoded = None

        if decoded and decoded.strip() and all(ch in string.printable for ch in decoded):
            return decoded.strip()

        return value.hex()

    def _is_printable(self, value: bytes) -> bool:
        """Return True if every character in ``value`` is human-readable."""
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError:
            return False

        cleaned = decoded.strip()
        if not cleaned:
            return False

        return all(ch in string.printable for ch in cleaned)

    def _is_numeric(self, value: bytes) -> bool:
        """Return True if ``value`` represents an integer string."""
        try:
            decoded = value.decode("utf-8").strip()
        except UnicodeDecodeError:
            return False

        if not decoded:
            return False

        if decoded.startswith(("+", "-")):
            decoded = decoded[1:]

        return decoded.isdigit()

    async def _analyze_structure(self, messages: List[bytes]) -> StructuralFeatures:
        """Analyze structural features of messages."""
        self.logger.debug("Analyzing structural features")

        # Length distribution
        length_dist = Counter(len(msg) for msg in messages)

        # Fixed/variable length detection
        unique_lengths = len(length_dist)
        is_fixed_length = unique_lengths == 1
        is_variable_length = unique_lengths > len(messages) * 0.1

        # Common prefixes and suffixes
        common_prefixes = await self._find_common_prefixes(messages)
        common_suffixes = await self._find_common_suffixes(messages)

        # Repeating sequences
        repeating_sequences = await self._find_repeating_sequences(messages)

        # Header/footer detection
        header_length = self._detect_header_length(common_prefixes)
        footer_length = self._detect_footer_length(common_suffixes)

        # Message classification
        message_classes = await self._classify_message_types(messages)

        return StructuralFeatures(
            length_distribution=dict(length_dist),
            common_prefixes=common_prefixes,
            common_suffixes=common_suffixes,
            repeating_sequences=repeating_sequences,
            field_boundaries=[],  # Will be filled by boundary detection
            header_length=header_length,
            footer_length=footer_length,
            is_fixed_length=is_fixed_length,
            is_variable_length=is_variable_length,
            message_classes=message_classes,
        )

    async def _detect_patterns(self, messages: List[bytes]) -> List[PatternInfo]:
        """Detect significant patterns in messages."""
        self.logger.debug("Detecting patterns")

        patterns = []
        pattern_candidates = defaultdict(list)

        # Extract all possible subsequences
        for msg_idx, message in enumerate(messages):
            for length in range(
                self.min_pattern_length, min(self.max_pattern_length + 1, len(message))
            ):
                for start in range(len(message) - length + 1):
                    pattern = message[start : start + length]
                    pattern_candidates[pattern].append((msg_idx, start))

        # Filter significant patterns
        for pattern, occurrences in pattern_candidates.items():
            if len(occurrences) >= self.min_pattern_frequency:
                # Calculate pattern statistics
                positions = [pos for _, pos in occurrences]
                contexts = []

                # Extract contexts (surrounding bytes)
                for msg_idx, pos in occurrences:
                    msg = messages[msg_idx]
                    context_start = max(0, pos - 8)
                    context_end = min(len(msg), pos + len(pattern) + 8)
                    contexts.append(msg[context_start:context_end])

                # Calculate pattern entropy
                pattern_entropy = entropy(
                    [pattern.count(b) for b in set(pattern)], base=2
                )

                # Determine pattern type
                pattern_type = self._classify_pattern_type(pattern, positions, contexts)

                # Calculate significance score
                significance = self._calculate_pattern_significance(
                    pattern, occurrences, messages
                )

                pattern_info = PatternInfo(
                    pattern=pattern,
                    frequency=len(occurrences),
                    positions=positions,
                    contexts=contexts,
                    pattern_type=pattern_type,
                    entropy=pattern_entropy,
                    significance=significance,
                )

                patterns.append(pattern_info)

        # Sort by significance
        patterns.sort(key=lambda p: p.significance, reverse=True)

        self.logger.debug(f"Detected {len(patterns)} significant patterns")
        return patterns[:100]  # Return top 100 patterns

    async def _detect_field_boundaries(
        self, messages: List[bytes]
    ) -> List[FieldBoundary]:
        """Detect potential field boundaries using multiple techniques."""
        self.logger.debug("Detecting field boundaries")

        if not messages:
            return []

        # Use cache if available
        cache_key = hashlib.md5(
            b"".join(messages[:10])
        ).hexdigest()  # Cache based on first 10 messages
        if cache_key in self._boundary_cache:
            return self._boundary_cache[cache_key]

        boundaries = []

        # Method 1: Entropy change detection
        entropy_boundaries = await self._detect_entropy_boundaries(messages)
        boundaries.extend(entropy_boundaries)

        # Method 2: Delimiter detection
        delimiter_boundaries = await self._detect_delimiter_boundaries(messages)
        boundaries.extend(delimiter_boundaries)

        # Method 3: Length field detection
        length_field_boundaries = await self._detect_length_field_boundaries(messages)
        boundaries.extend(length_field_boundaries)

        # Method 4: Pattern break detection
        pattern_break_boundaries = await self._detect_pattern_break_boundaries(messages)
        boundaries.extend(pattern_break_boundaries)

        # Merge and filter boundaries
        merged_boundaries = self._merge_boundaries(boundaries)
        filtered_boundaries = [
            b
            for b in merged_boundaries
            if b.confidence >= self.boundary_confidence_threshold
        ]

        # Cache results
        self._boundary_cache[cache_key] = filtered_boundaries

        self.logger.debug(f"Detected {len(filtered_boundaries)} field boundaries")
        return filtered_boundaries

    async def _classify_messages(self, messages: List[bytes]) -> Dict[str, Any]:
        """Classify messages into different types and categories."""
        self.logger.debug("Classifying messages")

        classifications = {
            "by_length": defaultdict(list),
            "by_entropy": defaultdict(list),
            "by_content_type": defaultdict(list),
            "clusters": [],
        }

        # Classify by length
        for i, msg in enumerate(messages):
            length_class = self._classify_by_length(len(msg))
            classifications["by_length"][length_class].append(i)

        # Classify by entropy
        for i, msg in enumerate(messages):
            entropy_val = entropy([msg.count(b) for b in set(msg)], base=2)
            entropy_class = self._classify_by_entropy(entropy_val)
            classifications["by_entropy"][entropy_class].append(i)

        # Classify by content type
        for i, msg in enumerate(messages):
            content_type = self._classify_content_type(msg)
            classifications["by_content_type"][content_type].append(i)

        # Cluster messages using feature vectors
        if len(messages) > 3:
            clusters = await self._cluster_messages(messages)
            classifications["clusters"] = clusters

        return dict(classifications)

    # Helper methods for pattern analysis

    def _is_likely_fixed_length(self, messages: List[bytes]) -> bool:
        """Determine if messages are likely fixed length."""
        if not messages:
            return False
        lengths = [len(msg) for msg in messages]
        return len(set(lengths)) <= max(1, len(messages) * 0.05)  # 5% tolerance

    def _is_likely_text(self, data: bytes) -> bool:
        """Determine if data is likely text content."""
        try:
            # Try to decode as UTF-8
            text = data.decode("utf-8")

            # Check for printable ASCII characters
            printable_count = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
            printable_ratio = printable_count / len(text) if text else 0

            return printable_ratio > 0.8
        except UnicodeDecodeError:
            # Check for common text bytes
            text_bytes = set(range(32, 127)) | {
                9,
                10,
                13,
            }  # Printable ASCII + tab, LF, CR
            text_count = sum(1 for b in data if b in text_bytes)
            return text_count / len(data) > 0.8 if data else False

    async def _find_common_prefixes(
        self, messages: List[bytes]
    ) -> List[Tuple[bytes, int]]:
        """Find common prefixes across messages."""
        if not messages:
            return []

        prefix_counts = defaultdict(int)
        min_length = min(len(msg) for msg in messages)

        for length in range(1, min(min_length + 1, 32)):
            for msg in messages:
                prefix = msg[:length]
                prefix_counts[prefix] += 1

        # Filter significant prefixes
        threshold = max(2, len(messages) * 0.1)
        significant_prefixes = [
            (prefix, count)
            for prefix, count in prefix_counts.items()
            if count >= threshold
        ]

        # Sort by count and length
        significant_prefixes.sort(key=lambda x: (x[1], len(x[0])), reverse=True)

        return significant_prefixes[:20]  # Return top 20

    async def _find_common_suffixes(
        self, messages: List[bytes]
    ) -> List[Tuple[bytes, int]]:
        """Find common suffixes across messages."""
        if not messages:
            return []

        suffix_counts = defaultdict(int)
        min_length = min(len(msg) for msg in messages)

        for length in range(1, min(min_length + 1, 32)):
            for msg in messages:
                suffix = msg[-length:]
                suffix_counts[suffix] += 1

        # Filter significant suffixes
        threshold = max(2, len(messages) * 0.1)
        significant_suffixes = [
            (suffix, count)
            for suffix, count in suffix_counts.items()
            if count >= threshold
        ]

        # Sort by count and length
        significant_suffixes.sort(key=lambda x: (x[1], len(x[0])), reverse=True)

        return significant_suffixes[:20]  # Return top 20

    async def _find_repeating_sequences(
        self, messages: List[bytes]
    ) -> List[PatternInfo]:
        """Find repeating sequences within individual messages."""
        repeating_patterns = []

        for msg in messages:
            patterns = self._find_repeating_in_message(msg)
            repeating_patterns.extend(patterns)

        # Aggregate patterns across messages
        pattern_map = defaultdict(list)
        for pattern in repeating_patterns:
            pattern_map[pattern.pattern].append(pattern)

        # Merge patterns
        merged_patterns = []
        for pattern_bytes, pattern_list in pattern_map.items():
            if len(pattern_list) >= 2:  # Pattern appears in multiple messages
                merged_frequency = sum(p.frequency for p in pattern_list)
                all_positions = []
                all_contexts = []

                for p in pattern_list:
                    all_positions.extend(p.positions)
                    all_contexts.extend(p.contexts)

                merged_pattern = PatternInfo(
                    pattern=pattern_bytes,
                    frequency=merged_frequency,
                    positions=all_positions,
                    contexts=all_contexts,
                    pattern_type="repeated",
                    entropy=entropy(
                        [pattern_bytes.count(b) for b in set(pattern_bytes)], base=2
                    ),
                    significance=merged_frequency / len(messages),
                )

                merged_patterns.append(merged_pattern)

        return merged_patterns

    def _find_repeating_in_message(self, message: bytes) -> List[PatternInfo]:
        """Find repeating patterns within a single message."""
        patterns = []

        for length in range(2, min(16, len(message) // 2)):
            for start in range(len(message) - length * 2 + 1):
                pattern = message[start : start + length]

                # Count consecutive repeats
                repeat_count = 0
                pos = start

                while (
                    pos + length <= len(message)
                    and message[pos : pos + length] == pattern
                ):
                    repeat_count += 1
                    pos += length

                if repeat_count >= 2:
                    patterns.append(
                        PatternInfo(
                            pattern=pattern,
                            frequency=repeat_count,
                            positions=[start + i * length for i in range(repeat_count)],
                            contexts=[
                                message[
                                    max(0, start - 4) : start
                                    + length * repeat_count
                                    + 4
                                ]
                            ],
                            pattern_type="repeated",
                            entropy=entropy(
                                [pattern.count(b) for b in set(pattern)], base=2
                            ),
                            significance=repeat_count * length / len(message),
                        )
                    )

        return patterns

    def _detect_header_length(
        self, common_prefixes: List[Tuple[bytes, int]]
    ) -> Optional[int]:
        """Detect likely header length from common prefixes."""
        if not common_prefixes:
            return None

        # Look for the longest common prefix with high frequency
        for prefix, count in common_prefixes:
            if count >= len(common_prefixes) * 0.8:  # 80% of messages
                return len(prefix)

        return None

    def _detect_footer_length(
        self, common_suffixes: List[Tuple[bytes, int]]
    ) -> Optional[int]:
        """Detect likely footer length from common suffixes."""
        if not common_suffixes:
            return None

        # Look for the longest common suffix with high frequency
        for suffix, count in common_suffixes:
            if count >= len(common_suffixes) * 0.8:  # 80% of messages
                return len(suffix)

        return None

    async def _classify_message_types(self, messages: List[bytes]) -> List[str]:
        """Classify message types based on content analysis."""
        message_types = []

        for msg in messages:
            msg_type = "unknown"

            # Basic classification rules
            if len(msg) == 0:
                msg_type = "empty"
            elif self._is_likely_text(msg):
                msg_type = "text"
            elif self._looks_like_binary_header(msg):
                msg_type = "binary_structured"
            elif self._looks_like_length_prefixed(msg):
                msg_type = "length_prefixed"
            elif self._looks_like_delimited(msg):
                msg_type = "delimited"
            else:
                msg_type = "binary_unstructured"

            message_types.append(msg_type)

        return message_types

    def _looks_like_binary_header(self, message: bytes) -> bool:
        """Check if message looks like it has a binary header."""
        if len(message) < 4:
            return False

        # Check for common binary header patterns
        header = message[:4]

        # Magic numbers or version indicators
        if header in [b"\x00\x01\x02\x03", b"\xff\xfe\xfd\xfc", b"\x01\x00\x00\x00"]:
            return True

        # Length fields (little/big endian)
        try:
            little_endian = struct.unpack("<I", header)[0]
            big_endian = struct.unpack(">I", header)[0]

            # Check if either interpretation gives a reasonable length
            if 4 <= little_endian <= len(message) or 4 <= big_endian <= len(message):
                return True
        except struct.error:
            pass

        return False

    def _looks_like_length_prefixed(self, message: bytes) -> bool:
        """Check if message appears to be length-prefixed."""
        if len(message) < 2:
            return False

        # Try different length field sizes
        for length_size in [1, 2, 4]:
            if len(message) < length_size:
                continue

            try:
                if length_size == 1:
                    length = message[0]
                elif length_size == 2:
                    length = struct.unpack(">H", message[:2])[0]
                else:
                    length = struct.unpack(">I", message[:4])[0]

                # Check if length matches remaining message
                if length + length_size == len(message):
                    return True

            except (struct.error, IndexError):
                continue

        return False

    def _looks_like_delimited(self, message: bytes) -> bool:
        """Check if message appears to use delimiters."""
        common_delimiters = [b"\x00", b"\x0a", b"\x0d", b"\x20", b"|", b",", b";"]

        for delimiter in common_delimiters:
            if delimiter in message and message.count(delimiter) >= 2:
                return True

        return False

    async def _detect_entropy_boundaries(
        self, messages: List[bytes]
    ) -> List[FieldBoundary]:
        """Detect boundaries based on entropy changes."""
        boundaries = []

        for msg in messages:
            if len(msg) < self.entropy_window_size * 2:
                continue

            # Calculate sliding window entropy
            entropies = []
            for i in range(len(msg) - self.entropy_window_size + 1):
                window = msg[i : i + self.entropy_window_size]
                window_entropy = entropy([window.count(b) for b in set(window)], base=2)
                entropies.append(window_entropy)

            # Detect significant entropy changes
            for i in range(1, len(entropies)):
                entropy_diff = abs(entropies[i] - entropies[i - 1])
                if entropy_diff > 1.0:  # Significant entropy change
                    confidence = min(entropy_diff / 4.0, 1.0)  # Normalize to 0-1

                    boundary = FieldBoundary(
                        position=i + self.entropy_window_size // 2,
                        confidence=confidence,
                        boundary_type="entropy_change",
                        evidence={
                            "entropy_before": entropies[i - 1],
                            "entropy_after": entropies[i],
                            "entropy_diff": entropy_diff,
                        },
                        separator=None,
                    )
                    boundaries.append(boundary)

        return boundaries

    async def _detect_delimiter_boundaries(
        self, messages: List[bytes]
    ) -> List[FieldBoundary]:
        """Detect boundaries based on delimiter patterns."""
        boundaries = []
        delimiter_candidates = defaultdict(list)

        # Find potential delimiters
        for msg_idx, msg in enumerate(messages):
            for pos, byte_val in enumerate(msg):
                delimiter_candidates[bytes([byte_val])].append((msg_idx, pos))

        # Evaluate delimiter candidates
        for delimiter, positions in delimiter_candidates.items():
            if (
                len(positions) >= len(messages) * 0.5
            ):  # Appears in at least 50% of messages
                # Check if delimiter appears at consistent relative positions
                relative_positions = []
                for msg_idx, pos in positions:
                    if messages[msg_idx]:
                        relative_pos = pos / len(messages[msg_idx])
                        relative_positions.append(relative_pos)

                # If positions are clustered, it's likely a structure delimiter
                if relative_positions:
                    std_dev = np.std(relative_positions)
                    if std_dev < 0.1:  # Low variance in relative positions
                        avg_pos = np.mean([pos for _, pos in positions])
                        confidence = min(len(positions) / len(messages), 1.0)

                        boundary = FieldBoundary(
                            position=int(avg_pos),
                            confidence=confidence,
                            boundary_type="delimiter",
                            evidence={
                                "delimiter": delimiter,
                                "frequency": len(positions),
                                "relative_position_std": std_dev,
                            },
                            separator=delimiter,
                        )
                        boundaries.append(boundary)

        return boundaries

    async def _detect_length_field_boundaries(
        self, messages: List[bytes]
    ) -> List[FieldBoundary]:
        """Detect boundaries based on length field patterns."""
        boundaries = []

        for msg in messages:
            # Check for potential length fields at the beginning
            for length_size in [1, 2, 4]:
                if (
                    len(msg) < length_size + 4
                ):  # Need at least some data after length field
                    continue

                try:
                    if length_size == 1:
                        declared_length = msg[0]
                    elif length_size == 2:
                        declared_length = struct.unpack(">H", msg[:2])[0]
                    else:
                        declared_length = struct.unpack(">I", msg[:4])[0]

                    # Check if declared length makes sense
                    remaining_length = len(msg) - length_size
                    if declared_length == remaining_length:
                        confidence = 0.9  # High confidence for exact match
                    else:
                        # Lower confidence if lengths don't match exactly
                        confidence = 0.5
                    boundary = FieldBoundary(
                        position=length_size,
                        confidence=confidence,
                        boundary_type="length_field",
                        evidence={
                            "length_field_size": length_size,
                            "declared_length": declared_length,
                            "actual_length": remaining_length,
                        },
                        separator=None,
                    )
                    boundaries.append(boundary)

                except (struct.error, IndexError):
                    continue

        return boundaries

    async def _detect_pattern_break_boundaries(
        self, messages: List[bytes]
    ) -> List[FieldBoundary]:
        """Detect boundaries where patterns change significantly."""
        boundaries = []

        # This is a simplified implementation
        # In practice, you'd use more sophisticated pattern analysis

        for msg in messages:
            if len(msg) < 8:
                continue

            # Look for transitions between different byte value ranges
            transitions = []
            for i in range(len(msg) - 1):
                curr_byte = msg[i]
                next_byte = msg[i + 1]

                # Detect transitions between different value ranges
                if (curr_byte < 32 and next_byte >= 32) or (
                    curr_byte >= 32 and next_byte < 32
                ):
                    transitions.append(i + 1)
                elif abs(int(curr_byte) - int(next_byte)) > 128:
                    transitions.append(i + 1)

            # Convert significant transitions to boundaries
            for pos in transitions:
                boundary = FieldBoundary(
                    position=pos,
                    confidence=0.5,  # Medium confidence for pattern breaks
                    boundary_type="pattern_break",
                    evidence={
                        "byte_before": msg[pos - 1] if pos > 0 else None,
                        "byte_after": msg[pos] if pos < len(msg) else None,
                    },
                    separator=None,
                )
                boundaries.append(boundary)

        return boundaries

    def _merge_boundaries(self, boundaries: List[FieldBoundary]) -> List[FieldBoundary]:
        """Merge nearby boundaries and resolve conflicts."""
        if not boundaries:
            return []

        # Sort by position
        boundaries.sort(key=lambda b: b.position)

        merged = []
        current_boundary = boundaries[0]

        for next_boundary in boundaries[1:]:
            # If boundaries are close (within 2 bytes), merge them
            if abs(next_boundary.position - current_boundary.position) <= 2:
                # Keep the boundary with higher confidence
                if next_boundary.confidence > current_boundary.confidence:
                    current_boundary = next_boundary
            else:
                merged.append(current_boundary)
                current_boundary = next_boundary

        merged.append(current_boundary)
        return merged

    async def detect_field_boundaries(self, messages: List[bytes]) -> List[FieldBoundary]:
        """Public interface for field boundary detection."""
        boundaries = await self._detect_field_boundaries(messages)
        # Ensure delimiter boundaries expose separator attribute for consumers
        for boundary in boundaries:
            if boundary.separator is None and "delimiter" in boundary.evidence:
                delimiter_value = boundary.evidence.get("delimiter")
                if isinstance(delimiter_value, (bytes, bytearray)):
                    boundary.separator = bytes(delimiter_value)
        return boundaries

    async def detect_patterns(self, messages: List[bytes]) -> List[PatternInfo]:
        """Public interface for pattern detection."""
        if not messages:
            return []
        patterns = await self._detect_patterns(messages)
        return patterns

    def _classify_pattern_type(
        self, pattern: bytes, positions: List[int], contexts: List[bytes]
    ) -> str:
        """Classify the type of pattern detected."""
        if len(set(pattern)) == 1:
            return "fixed"
        elif len(pattern) <= 4 and all(p % len(pattern) == 0 for p in positions):
            return "repeated"
        elif self._is_alternating_pattern(pattern):
            return "alternating"
        else:
            return "structured"

    def _is_alternating_pattern(self, pattern: bytes) -> bool:
        """Check if pattern alternates between two values."""
        if len(pattern) < 2:
            return False

        unique_bytes = list(set(pattern))
        if len(unique_bytes) != 2:
            return False

        # Check if pattern alternates
        for i in range(len(pattern) - 1):
            if pattern[i] == pattern[i + 1]:
                return False

        return True

    def _calculate_pattern_significance(
        self, pattern: bytes, occurrences: List[Tuple[int, int]], messages: List[bytes]
    ) -> float:
        """Calculate statistical significance of a pattern."""
        pattern_length = len(pattern)
        total_bytes = sum(len(msg) for msg in messages)

        # Expected frequency under null hypothesis (random occurrence)
        expected_freq = total_bytes / (256**pattern_length)

        # Actual frequency
        actual_freq = len(occurrences)

        # Significance score (log ratio)
        if expected_freq > 0:
            significance = np.log2(actual_freq / expected_freq)
        else:
            significance = np.log2(actual_freq + 1)

        # Normalize by pattern length (longer patterns are more significant)
        significance *= pattern_length

        return max(0.0, significance)

    def _classify_by_length(self, length: int) -> str:
        """Classify message by length."""
        if length == 0:
            return "empty"
        elif length <= 16:
            return "very_short"
        elif length <= 64:
            return "short"
        elif length <= 256:
            return "medium"
        elif length <= 1024:
            return "long"
        else:
            return "very_long"

    def _classify_by_entropy(self, entropy_val: float) -> str:
        """Classify message by entropy."""
        if entropy_val < 2.0:
            return "very_low_entropy"
        elif entropy_val < 4.0:
            return "low_entropy"
        elif entropy_val < 6.0:
            return "medium_entropy"
        elif entropy_val < 7.5:
            return "high_entropy"
        else:
            return "very_high_entropy"

    def _classify_content_type(self, message: bytes) -> str:
        """Classify message content type."""
        if not message:
            return "empty"
        elif self._is_likely_text(message):
            return "text"
        elif self._looks_like_binary_header(message):
            return "binary_structured"
        elif message[0] == 0 or message[-1] == 0:
            return "null_terminated"
        else:
            return "binary_raw"

    async def _cluster_messages(self, messages: List[bytes]) -> List[Dict[str, Any]]:
        """Cluster messages based on statistical features."""
        if len(messages) < 4:
            return []

        # Extract features for clustering
        features = []
        for msg in messages:
            feature_vector = [
                len(msg),  # Length
                entropy([msg.count(b) for b in set(msg)], base=2),  # Entropy
                len(set(msg)),  # Unique byte count
                msg.count(0) / len(msg) if msg else 0,  # Null byte ratio
                (
                    sum(1 for b in msg if 32 <= b <= 126) / len(msg) if msg else 0
                ),  # Printable ratio
            ]
            features.append(feature_vector)

        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        # Perform DBSCAN clustering
        clusterer = DBSCAN(
            eps=self.clustering_eps, min_samples=self.clustering_min_samples
        )
        cluster_labels = clusterer.fit_predict(normalized_features)

        # Organize results
        clusters = []
        for label in set(cluster_labels):
            if label == -1:  # Noise points
                continue

            cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
            cluster_info = {
                "cluster_id": int(label),
                "message_indices": cluster_indices,
                "size": len(cluster_indices),
                "representative_features": np.mean(
                    [features[i] for i in cluster_indices], axis=0
                ).tolist(),
            }
            clusters.append(cluster_info)

        return clusters

    async def shutdown(self):
        """Shutdown the statistical analyzer and cleanup resources."""
        self.logger.info("Shutting down Statistical Analyzer")

        if self.executor:
            self.executor.shutdown(wait=True)

        # Clear caches
        self._pattern_cache.clear()
        self._boundary_cache.clear()

        self.logger.info("Statistical Analyzer shutdown completed")
