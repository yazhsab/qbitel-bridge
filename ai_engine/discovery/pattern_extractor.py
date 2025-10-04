"""CRONOS AI Engine - Pattern Extractor (lightweight implementation).

The extractor surfaces repeated byte patterns along with frequency and
statistical metadata that can be consumed by downstream modules and the
unit tests that exercise protocol discovery capabilities.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ..core.config import Config


@dataclass
class PatternResult:
    """Information about an extracted pattern."""

    pattern: bytes
    frequency: int
    coverage: float
    average_gap: float
    contexts: List[bytes] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


class PatternExtractor:
    """Simple byte-pattern extractor suitable for protocol discovery tests."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.min_length = 2
        self.max_length = 6
        self.min_frequency = 2

    async def extract(self, messages: Iterable[bytes]) -> List[PatternResult]:
        """Extract repeated byte patterns from the supplied messages."""
        pattern_occurrences: Dict[bytes, List[int]] = defaultdict(list)
        total_bytes = 0

        for message in messages:
            if not message:
                continue
            total_bytes += len(message)
            for length in range(self.min_length, min(self.max_length, len(message)) + 1):
                for index in range(len(message) - length + 1):
                    pattern = message[index : index + length]
                    pattern_occurrences[pattern].append(index)

        # Filter by frequency threshold
        frequent_patterns = {
            pattern: indices
            for pattern, indices in pattern_occurrences.items()
            if len(indices) >= self.min_frequency
        }

        results: List[PatternResult] = []
        for pattern, indices in frequent_patterns.items():
            indices.sort()
            gaps = [j - i for i, j in zip(indices, indices[1:])] or [0]
            coverage = (len(pattern) * len(indices)) / max(total_bytes, 1)
            contexts = [pattern]  # simple placeholder for surrounding bytes

            metadata = {
                "first_occurrence": indices[0],
                "last_occurrence": indices[-1],
                "occurrence_positions": indices,
            }

            results.append(
                PatternResult(
                    pattern=pattern,
                    frequency=len(indices),
                    coverage=float(coverage),
                    average_gap=float(sum(gaps) / len(gaps)),
                    contexts=contexts,
                    metadata=metadata,
                )
            )

        # Rank results by frequency then coverage
        results.sort(key=lambda r: (r.frequency, r.coverage), reverse=True)
        return results[:100]


__all__ = ["PatternExtractor", "PatternResult"]
