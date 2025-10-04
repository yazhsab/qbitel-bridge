"""
CRONOS AI Engine - PCFG (Probabilistic Context-Free Grammar) Inference

This module implements advanced PCFG inference for automatic protocol grammar
learning from network traffic samples.
"""

import logging
import math
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from scipy.stats import entropy
import networkx as nx

from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException


@dataclass
class ProductionRule:
    """Represents a PCFG production rule."""

    left_hand_side: str  # Non-terminal symbol
    right_hand_side: List[str]  # Sequence of terminals and non-terminals
    probability: float
    frequency: int = 0
    contexts: List[str] = None  # Contexts where this rule appears

    def __post_init__(self):
        if self.contexts is None:
            self.contexts = []

    def __str__(self) -> str:
        rhs = " ".join(self.right_hand_side)
        return f"{self.left_hand_side} -> {rhs} [{self.probability:.4f}]"

    def is_terminal_rule(self) -> bool:
        """Check if this is a terminal production rule."""
        return all(not symbol.startswith("<") for symbol in self.right_hand_side)


@dataclass
class Grammar:
    """Represents a complete PCFG grammar."""

    rules: List[ProductionRule]
    terminals: Set[str]
    non_terminals: Set[str]
    start_symbol: str = "<START>"

    def __post_init__(self):
        self._rule_index = self._build_rule_index()

    def _build_rule_index(self) -> Dict[str, List[ProductionRule]]:
        """Build an index of rules by left-hand side."""
        index = defaultdict(list)
        for rule in self.rules:
            index[rule.left_hand_side].append(rule)
        return index

    def get_rules_for_symbol(self, symbol: str) -> List[ProductionRule]:
        """Get all production rules for a given non-terminal symbol."""
        return self._rule_index.get(symbol, [])

    def calculate_grammar_complexity(self) -> float:
        """Calculate the complexity of the grammar."""
        return len(self.rules) + len(self.non_terminals) * math.log(
            len(self.terminals) + 1
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert grammar to dictionary representation."""
        return {
            "rules": [
                {
                    "lhs": rule.left_hand_side,
                    "rhs": rule.right_hand_side,
                    "probability": rule.probability,
                    "frequency": rule.frequency,
                }
                for rule in self.rules
            ],
            "terminals": list(self.terminals),
            "non_terminals": list(self.non_terminals),
            "start_symbol": self.start_symbol,
        }


class PCFGInference:
    """
    PCFG Inference Engine for automatic protocol grammar learning.

    This class implements advanced algorithms for inferring probabilistic
    context-free grammars from protocol message samples.
    """

    def __init__(self, config: Config):
        """Initialize PCFG inference engine."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Algorithm parameters
        self.min_pattern_frequency = 3
        self.max_rule_length = 10
        self.min_symbol_entropy = 0.5
        self.max_grammar_size = 1000
        self.convergence_threshold = 0.001
        self.max_iterations = 100

        # State variables
        self._message_samples: List[bytes] = []
        self._byte_frequencies: Dict[int, int] = defaultdict(int)
        self._pattern_cache: Dict[str, Any] = {}

    async def infer(self, message_samples: List[bytes]) -> Grammar:
        """
        Infer PCFG grammar from message samples.

        Args:
            message_samples: List of protocol message samples

        Returns:
            Inferred PCFG grammar
        """
        if not message_samples:
            raise ProtocolException("Empty message samples provided")

        self.logger.info(f"Starting PCFG inference on {len(message_samples)} samples")

        try:
            # Store samples for analysis
            self._message_samples = message_samples

            # Step 1: Preprocess and tokenize messages
            tokens = self._tokenize_messages(message_samples)

            # Step 2: Extract frequent patterns
            patterns = self._extract_frequent_patterns(tokens)

            # Step 3: Identify terminals and non-terminals
            terminals, non_terminals = self._identify_symbols(patterns)

            # Step 4: Generate initial production rules
            initial_rules = self._generate_initial_rules(
                patterns, terminals, non_terminals
            )

            # Step 5: Refine grammar using EM algorithm
            refined_rules = await self._refine_grammar_em(initial_rules, tokens)

            # Step 6: Build final grammar
            grammar = Grammar(
                rules=refined_rules, terminals=terminals, non_terminals=non_terminals
            )

            self.logger.info(
                f"PCFG inference completed: {len(refined_rules)} rules, "
                f"{len(terminals)} terminals, {len(non_terminals)} non-terminals"
            )

            return grammar

        except Exception as e:
            self.logger.error(f"PCFG inference failed: {e}")
            raise ModelException(f"PCFG inference error: {e}")

    def _tokenize_messages(self, messages: List[bytes]) -> List[List[str]]:
        """
        Tokenize messages into sequences of symbols.

        Args:
            messages: Raw message bytes

        Returns:
            List of tokenized message sequences
        """
        self.logger.debug("Tokenizing messages")

        tokenized_messages = []

        for msg in messages:
            # Convert to hex representation for easier pattern matching
            hex_msg = msg.hex()

            # Tokenize based on different strategies
            tokens = []

            # Strategy 1: Byte-level tokenization
            byte_tokens = [hex_msg[i : i + 2] for i in range(0, len(hex_msg), 2)]

            # Strategy 2: Identify potential delimiters and separators
            delimiter_tokens = self._identify_delimiters(byte_tokens)

            # Strategy 3: Identify repeating patterns
            pattern_tokens = self._identify_repeating_patterns(byte_tokens)

            # Combine strategies
            tokens = self._merge_tokenization_strategies(
                byte_tokens, delimiter_tokens, pattern_tokens
            )

            tokenized_messages.append(tokens)

            # Update byte frequency statistics
            for token in byte_tokens:
                if len(token) == 2:  # Single byte token
                    self._byte_frequencies[int(token, 16)] += 1

        return tokenized_messages

    def _identify_delimiters(self, tokens: List[str]) -> List[str]:
        """Identify potential delimiter tokens."""
        # Common delimiter patterns in protocols
        delimiter_patterns = [
            r"00+",  # Null byte sequences
            r"ff+",  # 0xFF sequences
            r"0d0a",  # CRLF
            r"20+",  # Space sequences
        ]

        result_tokens = []
        i = 0

        while i < len(tokens):
            token = tokens[i]
            merged = False

            # Check for delimiter patterns
            for pattern in delimiter_patterns:
                if re.match(pattern, token):
                    # Try to extend the delimiter
                    delimiter_seq = [token]
                    j = i + 1
                    while j < len(tokens) and re.match(pattern, tokens[j]):
                        delimiter_seq.append(tokens[j])
                        j += 1

                    if len(delimiter_seq) > 1:
                        # Merge delimiter sequence
                        result_tokens.append("<DELIM_" + "_".join(delimiter_seq) + ">")
                        i = j
                        merged = True
                        break

            if not merged:
                result_tokens.append(token)
                i += 1

        return result_tokens

    def _identify_repeating_patterns(self, tokens: List[str]) -> List[str]:
        """Identify repeating patterns in token sequences."""
        result_tokens = tokens.copy()

        # Look for repeating subsequences
        for length in range(2, min(6, len(tokens) // 2)):
            i = 0
            while i <= len(result_tokens) - length * 2:
                pattern = result_tokens[i : i + length]

                # Check if pattern repeats
                repeat_count = 1
                j = i + length
                while j <= len(result_tokens) - length:
                    if result_tokens[j : j + length] == pattern:
                        repeat_count += 1
                        j += length
                    else:
                        break

                if repeat_count >= 2:
                    # Replace repeating pattern with a non-terminal
                    pattern_name = f"<REPEAT_{length}_{repeat_count}>"
                    new_tokens = result_tokens[:i] + [pattern_name] + result_tokens[j:]
                    result_tokens = new_tokens
                else:
                    i += 1

        return result_tokens

    def _merge_tokenization_strategies(
        self,
        byte_tokens: List[str],
        delimiter_tokens: List[str],
        pattern_tokens: List[str],
    ) -> List[str]:
        """Merge different tokenization strategies."""
        # For now, use the most informative tokenization
        # In future, could use ensemble approach

        # Choose based on information content
        byte_entropy = self._calculate_sequence_entropy(byte_tokens)
        delimiter_entropy = self._calculate_sequence_entropy(delimiter_tokens)
        pattern_entropy = self._calculate_sequence_entropy(pattern_tokens)

        if pattern_entropy > delimiter_entropy and pattern_entropy > byte_entropy:
            return pattern_tokens
        elif delimiter_entropy > byte_entropy:
            return delimiter_tokens
        else:
            return byte_tokens

    def _calculate_sequence_entropy(self, sequence: List[str]) -> float:
        """Calculate entropy of a token sequence."""
        if not sequence:
            return 0.0

        counts = Counter(sequence)
        total = len(sequence)
        probabilities = [count / total for count in counts.values()]
        return entropy(probabilities, base=2)

    def _extract_frequent_patterns(
        self, tokenized_messages: List[List[str]]
    ) -> Dict[str, Any]:
        """Extract frequent patterns from tokenized messages."""
        self.logger.debug("Extracting frequent patterns")

        patterns = {
            "unigrams": Counter(),
            "bigrams": Counter(),
            "trigrams": Counter(),
            "common_subsequences": {},
            "structural_patterns": [],
        }

        # Extract n-grams
        for tokens in tokenized_messages:
            # Unigrams
            for token in tokens:
                patterns["unigrams"][token] += 1

            # Bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                patterns["bigrams"][bigram] += 1

            # Trigrams
            for i in range(len(tokens) - 2):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                patterns["trigrams"][trigram] += 1

        # Extract common subsequences using suffix array approach
        patterns["common_subsequences"] = self._find_common_subsequences(
            tokenized_messages
        )

        # Extract structural patterns
        patterns["structural_patterns"] = self._extract_structural_patterns(
            tokenized_messages
        )

        return patterns

    def _find_common_subsequences(
        self, tokenized_messages: List[List[str]]
    ) -> Dict[Tuple[str, ...], int]:
        """Find common subsequences across messages."""
        subsequence_counts = Counter()

        # Use sliding window approach
        for tokens in tokenized_messages:
            for length in range(2, min(self.max_rule_length, len(tokens) + 1)):
                for i in range(len(tokens) - length + 1):
                    subseq = tuple(tokens[i : i + length])
                    subsequence_counts[subseq] += 1

        # Filter by minimum frequency
        common_subsequences = {
            subseq: count
            for subseq, count in subsequence_counts.items()
            if count >= self.min_pattern_frequency
        }

        return common_subsequences

    def _extract_structural_patterns(
        self, tokenized_messages: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """Extract structural patterns like headers, bodies, footers."""
        structural_patterns = []

        if not tokenized_messages:
            return structural_patterns

        # Analyze message structure consistency
        lengths = [len(msg) for msg in tokenized_messages]
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)

        # Fixed-length messages
        if std_length < 0.1 * avg_length:
            structural_patterns.append(
                {
                    "type": "fixed_length",
                    "length": int(avg_length),
                    "confidence": 1.0 - std_length / avg_length,
                }
            )

        # Common prefix/suffix analysis
        if len(tokenized_messages) > 1:
            # Find longest common prefix
            prefix_len = 0
            for i in range(min(len(msg) for msg in tokenized_messages)):
                if all(
                    msg[i] == tokenized_messages[0][i] for msg in tokenized_messages[1:]
                ):
                    prefix_len += 1
                else:
                    break

            if prefix_len > 0:
                structural_patterns.append(
                    {
                        "type": "common_prefix",
                        "length": prefix_len,
                        "pattern": tokenized_messages[0][:prefix_len],
                        "confidence": 1.0,
                    }
                )

            # Find longest common suffix
            suffix_len = 0
            min_len = min(len(msg) for msg in tokenized_messages)
            for i in range(1, min_len + 1):
                if all(
                    msg[-i] == tokenized_messages[0][-i]
                    for msg in tokenized_messages[1:]
                ):
                    suffix_len += 1
                else:
                    break

            if suffix_len > 0:
                structural_patterns.append(
                    {
                        "type": "common_suffix",
                        "length": suffix_len,
                        "pattern": tokenized_messages[0][-suffix_len:],
                        "confidence": 1.0,
                    }
                )

        return structural_patterns

    def _identify_symbols(self, patterns: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
        """Identify terminal and non-terminal symbols."""
        self.logger.debug("Identifying terminal and non-terminal symbols")

        terminals = set()
        non_terminals = set()

        # Terminals: frequent unigrams that appear as single tokens
        for token, freq in patterns["unigrams"].items():
            if freq >= self.min_pattern_frequency:
                # Check if it's a simple byte token
                if len(token) <= 4 and not token.startswith("<"):
                    terminals.add(token)
                elif token.startswith("<"):
                    non_terminals.add(token)

        # Create non-terminals for frequent patterns
        for subseq, freq in patterns["common_subsequences"].items():
            if freq >= self.min_pattern_frequency and len(subseq) > 1:
                nt_name = f"<PATTERN_{len(subseq)}_{hash(subseq) % 10000}>"
                non_terminals.add(nt_name)

                # Add component terminals
                for token in subseq:
                    if not token.startswith("<"):
                        terminals.add(token)

        # Add structural non-terminals
        for pattern in patterns["structural_patterns"]:
            if pattern["type"] == "common_prefix":
                non_terminals.add("<HEADER>")
            elif pattern["type"] == "common_suffix":
                non_terminals.add("<FOOTER>")

        # Ensure we have a start symbol
        non_terminals.add("<START>")
        non_terminals.add("<MESSAGE>")
        non_terminals.add("<BODY>")

        self.logger.debug(
            f"Identified {len(terminals)} terminals and {len(non_terminals)} non-terminals"
        )

        return terminals, non_terminals

    def _generate_initial_rules(
        self, patterns: Dict[str, Any], terminals: Set[str], non_terminals: Set[str]
    ) -> List[ProductionRule]:
        """Generate initial production rules from patterns."""
        self.logger.debug("Generating initial production rules")

        rules = []

        # Start rule
        rules.append(
            ProductionRule(
                left_hand_side="<START>",
                right_hand_side=["<MESSAGE>"],
                probability=1.0,
                frequency=len(self._message_samples),
            )
        )

        # Structural rules
        has_header = any(
            p["type"] == "common_prefix" for p in patterns["structural_patterns"]
        )
        has_footer = any(
            p["type"] == "common_suffix" for p in patterns["structural_patterns"]
        )

        if has_header and has_footer:
            rules.append(
                ProductionRule(
                    left_hand_side="<MESSAGE>",
                    right_hand_side=["<HEADER>", "<BODY>", "<FOOTER>"],
                    probability=0.6,
                    frequency=int(len(self._message_samples) * 0.6),
                )
            )
        elif has_header:
            rules.append(
                ProductionRule(
                    left_hand_side="<MESSAGE>",
                    right_hand_side=["<HEADER>", "<BODY>"],
                    probability=0.7,
                    frequency=int(len(self._message_samples) * 0.7),
                )
            )
        elif has_footer:
            rules.append(
                ProductionRule(
                    left_hand_side="<MESSAGE>",
                    right_hand_side=["<BODY>", "<FOOTER>"],
                    probability=0.7,
                    frequency=int(len(self._message_samples) * 0.7),
                )
            )
        else:
            rules.append(
                ProductionRule(
                    left_hand_side="<MESSAGE>",
                    right_hand_side=["<BODY>"],
                    probability=0.8,
                    frequency=int(len(self._message_samples) * 0.8),
                )
            )

        # Pattern-based rules
        for subseq, freq in patterns["common_subsequences"].items():
            if len(subseq) > 1:
                nt_name = f"<PATTERN_{len(subseq)}_{hash(subseq) % 10000}>"
                if nt_name in non_terminals:
                    rules.append(
                        ProductionRule(
                            left_hand_side=nt_name,
                            right_hand_side=list(subseq),
                            probability=freq / len(self._message_samples),
                            frequency=freq,
                        )
                    )

        # Terminal rules for frequent tokens
        body_alternatives = []
        for token, freq in patterns["unigrams"].most_common(20):
            if token in terminals:
                body_alternatives.append((token, freq))

        # Generate body production rules
        total_body_freq = sum(freq for _, freq in body_alternatives)
        for token, freq in body_alternatives:
            if total_body_freq > 0:
                rules.append(
                    ProductionRule(
                        left_hand_side="<BODY>",
                        right_hand_side=[token],
                        probability=freq / total_body_freq,
                        frequency=freq,
                    )
                )

        # Sequence rules for bodies
        for bigram, freq in patterns["bigrams"].most_common(10):
            if all(token in terminals or token in non_terminals for token in bigram):
                rules.append(
                    ProductionRule(
                        left_hand_side="<BODY>",
                        right_hand_side=list(bigram),
                        probability=freq / len(self._message_samples),
                        frequency=freq,
                    )
                )

        self.logger.debug(f"Generated {len(rules)} initial production rules")

        return rules

    async def _refine_grammar_em(
        self, initial_rules: List[ProductionRule], tokenized_messages: List[List[str]]
    ) -> List[ProductionRule]:
        """Refine grammar using Expectation-Maximization algorithm."""
        self.logger.debug("Refining grammar using EM algorithm")

        current_rules = initial_rules.copy()

        for iteration in range(self.max_iterations):
            self.logger.debug(f"EM iteration {iteration + 1}/{self.max_iterations}")

            # E-step: Calculate expected counts
            expected_counts = self._calculate_expected_counts(
                current_rules, tokenized_messages
            )

            # M-step: Update rule probabilities
            new_rules = self._update_rule_probabilities(current_rules, expected_counts)

            # Check convergence
            if self._has_converged(current_rules, new_rules):
                self.logger.debug(f"EM converged after {iteration + 1} iterations")
                break

            current_rules = new_rules

        # Filter low-probability rules
        filtered_rules = [
            rule
            for rule in current_rules
            if rule.probability > 0.001 or rule.frequency > self.min_pattern_frequency
        ]

        self.logger.debug(f"Refined to {len(filtered_rules)} production rules")

        return filtered_rules

    def _calculate_expected_counts(
        self, rules: List[ProductionRule], tokenized_messages: List[List[str]]
    ) -> Dict[str, float]:
        """Calculate expected counts for each rule (E-step)."""
        expected_counts = defaultdict(float)

        # Simple approximation: count actual occurrences in messages
        for tokens in tokenized_messages:
            for rule in rules:
                rhs = rule.right_hand_side

                if len(rhs) == 1:
                    # Terminal rule
                    count = tokens.count(rhs[0])
                    expected_counts[str(rule)] += count
                else:
                    # Multi-token rule
                    count = self._count_subsequence_occurrences(tokens, rhs)
                    expected_counts[str(rule)] += count

        return expected_counts

    def _count_subsequence_occurrences(
        self, tokens: List[str], pattern: List[str]
    ) -> int:
        """Count occurrences of a pattern in a token sequence."""
        count = 0
        for i in range(len(tokens) - len(pattern) + 1):
            if tokens[i : i + len(pattern)] == pattern:
                count += 1
        return count

    def _update_rule_probabilities(
        self, rules: List[ProductionRule], expected_counts: Dict[str, float]
    ) -> List[ProductionRule]:
        """Update rule probabilities (M-step)."""
        # Group rules by left-hand side
        rules_by_lhs = defaultdict(list)
        for rule in rules:
            rules_by_lhs[rule.left_hand_side].append(rule)

        updated_rules = []

        for lhs, lhs_rules in rules_by_lhs.items():
            # Calculate total expected count for this LHS
            total_count = sum(expected_counts.get(str(rule), 0) for rule in lhs_rules)

            if total_count > 0:
                # Update probabilities to sum to 1
                for rule in lhs_rules:
                    expected_count = expected_counts.get(str(rule), 0)
                    new_probability = expected_count / total_count

                    updated_rule = ProductionRule(
                        left_hand_side=rule.left_hand_side,
                        right_hand_side=rule.right_hand_side,
                        probability=new_probability,
                        frequency=int(expected_count),
                        contexts=rule.contexts,
                    )
                    updated_rules.append(updated_rule)
            else:
                # Keep original probabilities if no counts
                updated_rules.extend(lhs_rules)

        return updated_rules

    def _has_converged(
        self, old_rules: List[ProductionRule], new_rules: List[ProductionRule]
    ) -> bool:
        """Check if EM algorithm has converged."""
        if len(old_rules) != len(new_rules):
            return False

        # Create mapping by rule string representation
        old_probs = {str(rule): rule.probability for rule in old_rules}
        new_probs = {str(rule): rule.probability for rule in new_rules}

        # Check if probability changes are below threshold
        max_change = 0.0
        for rule_str in old_probs:
            if rule_str in new_probs:
                change = abs(old_probs[rule_str] - new_probs[rule_str])
                max_change = max(max_change, change)

        return max_change < self.convergence_threshold
