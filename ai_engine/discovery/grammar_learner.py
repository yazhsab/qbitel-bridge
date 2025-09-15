"""
CRONOS AI Engine - Grammar Learner for Protocol Discovery

This module implements advanced grammar learning algorithms including enhanced PCFG inference,
context-sensitive grammar generation, and probabilistic parsing for protocol structure discovery.
"""

import asyncio
import logging
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np
from scipy.stats import entropy
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import pickle
import json
import hashlib

from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException
from .statistical_analyzer import StatisticalAnalyzer, PatternInfo


@dataclass
class Symbol:
    """Represents a grammar symbol (terminal or non-terminal)."""
    name: str
    is_terminal: bool
    frequency: int = 0
    contexts: List[str] = field(default_factory=list)
    entropy: float = 0.0
    semantic_type: Optional[str] = None  # 'length', 'id', 'data', 'delimiter', etc.


@dataclass
class ProductionRule:
    """Enhanced production rule with semantic information."""
    left_hand_side: Symbol
    right_hand_side: List[Symbol]
    probability: float
    frequency: int = 0
    contexts: List[str] = field(default_factory=list)
    semantic_role: Optional[str] = None  # 'header', 'body', 'footer', 'field', etc.
    confidence: float = 0.0
    
    def __str__(self) -> str:
        rhs_str = " ".join([s.name for s in self.right_hand_side])
        return f"{self.left_hand_side.name} -> {rhs_str} [{self.probability:.4f}]"
    
    def is_recursive(self) -> bool:
        """Check if this rule is recursive."""
        return any(s.name == self.left_hand_side.name for s in self.right_hand_side if not s.is_terminal)
    
    def is_terminal_rule(self) -> bool:
        """Check if this rule produces only terminals."""
        return all(s.is_terminal for s in self.right_hand_side)


@dataclass
class Grammar:
    """Enhanced grammar representation with semantic annotations."""
    rules: List[ProductionRule]
    symbols: Dict[str, Symbol]
    start_symbol: str = "<START>"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._rule_index = self._build_rule_index()
        self._symbol_index = {symbol.name: symbol for symbol in self.symbols.values()}
    
    def _build_rule_index(self) -> Dict[str, List[ProductionRule]]:
        """Build index of rules by left-hand side symbol."""
        index = defaultdict(list)
        for rule in self.rules:
            index[rule.left_hand_side.name].append(rule)
        return dict(index)
    
    def get_rules_for_symbol(self, symbol_name: str) -> List[ProductionRule]:
        """Get all production rules for a given symbol."""
        return self._rule_index.get(symbol_name, [])
    
    def get_terminal_symbols(self) -> Set[str]:
        """Get all terminal symbols."""
        return {name for name, symbol in self.symbols.items() if symbol.is_terminal}
    
    def get_non_terminal_symbols(self) -> Set[str]:
        """Get all non-terminal symbols."""
        return {name for name, symbol in self.symbols.items() if not symbol.is_terminal}
    
    def calculate_complexity(self) -> float:
        """Calculate grammar complexity score."""
        terminal_count = len(self.get_terminal_symbols())
        non_terminal_count = len(self.get_non_terminal_symbols())
        rule_count = len(self.rules)
        
        # Complexity based on number of rules and symbol variety
        complexity = rule_count * np.log(terminal_count + 1) + non_terminal_count * 2
        return complexity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert grammar to dictionary representation."""
        return {
            "start_symbol": self.start_symbol,
            "rules": [
                {
                    "lhs": rule.left_hand_side.name,
                    "rhs": [s.name for s in rule.right_hand_side],
                    "probability": rule.probability,
                    "frequency": rule.frequency,
                    "semantic_role": rule.semantic_role,
                    "confidence": rule.confidence
                }
                for rule in self.rules
            ],
            "symbols": {
                name: {
                    "name": symbol.name,
                    "is_terminal": symbol.is_terminal,
                    "frequency": symbol.frequency,
                    "entropy": symbol.entropy,
                    "semantic_type": symbol.semantic_type
                }
                for name, symbol in self.symbols.items()
            },
            "metadata": self.metadata
        }


class GrammarLearner:
    """
    Advanced grammar learning engine for protocol discovery.
    
    This class implements state-of-the-art algorithms for inferring probabilistic
    context-free grammars from protocol message samples, with support for
    semantic annotation and context-sensitive parsing.
    """
    
    def __init__(self, config: Config):
        """Initialize the grammar learner."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Algorithm parameters
        self.min_pattern_frequency = 3
        self.max_rule_length = 12
        self.min_symbol_entropy = 0.1
        self.max_grammar_size = 500
        self.convergence_threshold = 0.001
        self.max_em_iterations = 50
        self.semantic_analysis_enabled = True
        
        # Performance settings
        self.use_parallel_processing = True
        self.max_workers = config.inference.num_workers if hasattr(config, 'inference') else 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Components
        self.statistical_analyzer = StatisticalAnalyzer(config)
        
        # State and caching
        self._message_cache: Dict[str, Any] = {}
        self._grammar_cache: Dict[str, Grammar] = {}
        self._symbol_registry: Dict[str, Symbol] = {}
        
        self.logger.info("Grammar Learner initialized")

    async def learn_grammar(
        self, 
        messages: List[bytes],
        protocol_hint: Optional[str] = None
    ) -> Grammar:
        """
        Learn a probabilistic context-free grammar from message samples.
        
        Args:
            messages: List of protocol message samples
            protocol_hint: Optional hint about the protocol type
            
        Returns:
            Learned grammar with production rules and probabilities
        """
        if not messages:
            raise ProtocolException("Empty message list provided for grammar learning")
        
        start_time = time.time()
        self.logger.info(f"Starting grammar learning on {len(messages)} messages")
        
        try:
            # Cache key for memoization
            cache_key = self._generate_cache_key(messages, protocol_hint)
            if cache_key in self._grammar_cache:
                self.logger.info("Returning cached grammar")
                return self._grammar_cache[cache_key]
            
            # Step 1: Statistical preprocessing
            self.logger.debug("Performing statistical analysis")
            stats_result = await self.statistical_analyzer.analyze_messages(messages)
            
            # Step 2: Tokenize messages
            self.logger.debug("Tokenizing messages")
            tokenized_messages = await self._tokenize_messages(messages, stats_result)
            
            # Step 3: Extract structural patterns
            self.logger.debug("Extracting structural patterns")
            structural_patterns = await self._extract_structural_patterns(
                tokenized_messages, stats_result
            )
            
            # Step 4: Identify symbols
            self.logger.debug("Identifying grammar symbols")
            symbols = await self._identify_symbols(
                tokenized_messages, structural_patterns, stats_result
            )
            
            # Step 5: Generate initial rules
            self.logger.debug("Generating initial production rules")
            initial_rules = await self._generate_initial_rules(
                tokenized_messages, symbols, structural_patterns
            )
            
            # Step 6: Refine grammar using advanced techniques
            self.logger.debug("Refining grammar with EM and semantic analysis")
            refined_rules = await self._refine_grammar(
                initial_rules, tokenized_messages, symbols, stats_result
            )
            
            # Step 7: Post-process and optimize
            self.logger.debug("Post-processing grammar")
            optimized_rules = await self._optimize_grammar(refined_rules, symbols)
            
            # Step 8: Build final grammar
            grammar = Grammar(
                rules=optimized_rules,
                symbols=symbols,
                start_symbol="<START>",
                metadata={
                    "learning_time": time.time() - start_time,
                    "message_count": len(messages),
                    "protocol_hint": protocol_hint,
                    "complexity": 0.0  # Will be calculated
                }
            )
            
            grammar.metadata["complexity"] = grammar.calculate_complexity()
            
            # Cache result
            self._grammar_cache[cache_key] = grammar
            
            self.logger.info(
                f"Grammar learning completed in {time.time() - start_time:.2f}s: "
                f"{len(optimized_rules)} rules, {len(symbols)} symbols"
            )
            
            return grammar
            
        except Exception as e:
            self.logger.error(f"Grammar learning failed: {e}")
            raise ModelException(f"Grammar learning error: {e}")

    async def _tokenize_messages(
        self, 
        messages: List[bytes], 
        stats_result: Dict[str, Any]
    ) -> List[List[str]]:
        """Advanced message tokenization using statistical insights."""
        tokenized_messages = []
        
        # Extract tokenization hints from statistical analysis
        patterns = stats_result.get('patterns', [])
        boundaries = stats_result.get('field_boundaries', [])
        
        for message in messages:
            tokens = await self._tokenize_single_message(message, patterns, boundaries)
            tokenized_messages.append(tokens)
        
        return tokenized_messages

    async def _tokenize_single_message(
        self, 
        message: bytes, 
        patterns: List[PatternInfo], 
        boundaries: List[Any]
    ) -> List[str]:
        """Tokenize a single message using multiple strategies."""
        if not message:
            return []
        
        # Strategy 1: Pattern-based tokenization
        pattern_tokens = self._tokenize_by_patterns(message, patterns)
        
        # Strategy 2: Boundary-based tokenization
        boundary_tokens = self._tokenize_by_boundaries(message, boundaries)
        
        # Strategy 3: Entropy-based tokenization
        entropy_tokens = self._tokenize_by_entropy(message)
        
        # Strategy 4: Semantic tokenization (for known protocol types)
        semantic_tokens = self._tokenize_semantically(message)
        
        # Select best tokenization strategy
        best_tokens = self._select_best_tokenization([
            pattern_tokens, boundary_tokens, entropy_tokens, semantic_tokens
        ])
        
        return best_tokens

    def _tokenize_by_patterns(self, message: bytes, patterns: List[PatternInfo]) -> List[str]:
        """Tokenize based on detected patterns."""
        tokens = []
        pos = 0
        
        # Sort patterns by position and significance
        pattern_positions = []
        for pattern in patterns:
            for position in pattern.positions:
                if position < len(message):
                    pattern_positions.append((position, pattern))
        
        pattern_positions.sort()
        
        for position, pattern in pattern_positions:
            if position >= pos:
                # Add tokens before pattern
                if position > pos:
                    pre_pattern = message[pos:position]
                    tokens.extend(self._simple_tokenize(pre_pattern))
                
                # Add pattern as token
                pattern_bytes = pattern.pattern
                if position + len(pattern_bytes) <= len(message):
                    if message[position:position + len(pattern_bytes)] == pattern_bytes:
                        tokens.append(f"<PATTERN_{pattern.pattern.hex()[:8]}>")
                        pos = position + len(pattern_bytes)
        
        # Add remaining bytes
        if pos < len(message):
            tokens.extend(self._simple_tokenize(message[pos:]))
        
        return tokens if tokens else self._simple_tokenize(message)

    def _tokenize_by_boundaries(self, message: bytes, boundaries: List[Any]) -> List[str]:
        """Tokenize based on detected field boundaries."""
        if not boundaries:
            return self._simple_tokenize(message)
        
        tokens = []
        positions = sorted([b.position for b in boundaries if hasattr(b, 'position')])
        positions = [0] + [p for p in positions if 0 < p < len(message)] + [len(message)]
        
        for i in range(len(positions) - 1):
            start, end = positions[i], positions[i + 1]
            field_data = message[start:end]
            
            if len(field_data) == 1:
                tokens.append(f"0x{field_data.hex()}")
            elif len(field_data) <= 4:
                tokens.append(f"<FIELD_{field_data.hex()}>")
            else:
                # Tokenize large fields further
                field_tokens = self._simple_tokenize(field_data)
                if len(field_tokens) > 8:
                    tokens.append(f"<LARGE_FIELD_{len(field_data)}>")
                else:
                    tokens.extend(field_tokens)
        
        return tokens

    def _tokenize_by_entropy(self, message: bytes, window_size: int = 4) -> List[str]:
        """Tokenize based on entropy changes."""
        if len(message) < window_size * 2:
            return self._simple_tokenize(message)
        
        # Calculate sliding window entropy
        entropies = []
        for i in range(len(message) - window_size + 1):
            window = message[i:i + window_size]
            window_entropy = entropy([window.count(b) for b in set(window)], base=2)
            entropies.append(window_entropy)
        
        # Find entropy change points
        change_points = [0]
        for i in range(1, len(entropies)):
            if abs(entropies[i] - entropies[i-1]) > 1.0:  # Significant entropy change
                change_points.append(i)
        change_points.append(len(message))
        
        # Tokenize based on change points
        tokens = []
        for i in range(len(change_points) - 1):
            start, end = change_points[i], change_points[i + 1]
            segment = message[start:end]
            
            if len(segment) <= 2:
                tokens.extend([f"0x{bytes([b]).hex()}" for b in segment])
            else:
                segment_entropy = entropies[min(start, len(entropies) - 1)]
                if segment_entropy > 6.0:
                    tokens.append(f"<HIGH_ENTROPY_{len(segment)}>")
                elif segment_entropy < 2.0:
                    tokens.append(f"<LOW_ENTROPY_{len(segment)}>")
                else:
                    tokens.extend(self._simple_tokenize(segment))
        
        return tokens

    def _tokenize_semantically(self, message: bytes) -> List[str]:
        """Tokenize based on semantic understanding of common protocol patterns."""
        tokens = []
        pos = 0
        
        # Check for common protocol headers
        if len(message) >= 4:
            header = message[:4]
            
            # Check for length-prefixed format
            try:
                # Try big-endian length
                length_be = int.from_bytes(header[:2], 'big')
                length_le = int.from_bytes(header[:2], 'little')
                
                if length_be + 2 == len(message):
                    tokens.extend(["<LENGTH_BE>", f"<DATA_{len(message)-2}>"])
                    return tokens
                elif length_le + 2 == len(message):
                    tokens.extend(["<LENGTH_LE>", f"<DATA_{len(message)-2}>"])
                    return tokens
                
                # Try 4-byte length
                length4_be = int.from_bytes(header, 'big')
                length4_le = int.from_bytes(header, 'little')
                
                if length4_be + 4 == len(message):
                    tokens.extend(["<LENGTH32_BE>", f"<DATA_{len(message)-4}>"])
                    return tokens
                elif length4_le + 4 == len(message):
                    tokens.extend(["<LENGTH32_LE>", f"<DATA_{len(message)-4}>"])
                    return tokens
                    
            except (ValueError, OverflowError):
                pass
        
        # Check for text-based protocols
        try:
            text = message.decode('utf-8')
            if all(c.isprintable() or c in '\r\n\t' for c in text):
                # Text protocol - split by common delimiters
                for delimiter in ['\r\n', '\n', '\t', ' ', '|']:
                    if delimiter in text:
                        parts = text.split(delimiter)
                        return [f"<TEXT_{i}>" for i, _ in enumerate(parts) if parts[i]]
                
                return ["<TEXT_DATA>"]
                
        except UnicodeDecodeError:
            pass
        
        # Default to simple tokenization
        return self._simple_tokenize(message)

    def _simple_tokenize(self, data: bytes) -> List[str]:
        """Simple byte-level tokenization."""
        if not data:
            return []
        
        # For short data, use hex representation
        if len(data) <= 8:
            return [f"0x{bytes([b]).hex()}" for b in data]
        
        # For longer data, group bytes
        tokens = []
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            if len(chunk) == 4:
                tokens.append(f"<BYTES_{chunk.hex()}>")
            else:
                tokens.extend([f"0x{bytes([b]).hex()}" for b in chunk])
        
        return tokens

    def _select_best_tokenization(self, tokenizations: List[List[str]]) -> List[str]:
        """Select the best tokenization strategy based on information content."""
        if not any(tokenizations):
            return []
        
        best_tokens = None
        best_score = -1
        
        for tokens in tokenizations:
            if not tokens:
                continue
            
            # Score based on token diversity and reasonable length
            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)
            
            if total_tokens == 0:
                score = 0
            else:
                diversity_score = unique_tokens / total_tokens
                length_penalty = 1.0 / (1.0 + abs(total_tokens - 10))  # Prefer ~10 tokens
                score = diversity_score * length_penalty
            
            if score > best_score:
                best_score = score
                best_tokens = tokens
        
        return best_tokens or tokenizations[0] if tokenizations[0] else []

    async def _extract_structural_patterns(
        self, 
        tokenized_messages: List[List[str]], 
        stats_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract high-level structural patterns from tokenized messages."""
        patterns = {
            'message_templates': [],
            'common_sequences': {},
            'positional_patterns': {},
            'semantic_structure': {}
        }
        
        # Extract message templates (common structure patterns)
        templates = await self._extract_message_templates(tokenized_messages)
        patterns['message_templates'] = templates
        
        # Extract common token sequences
        common_seqs = await self._extract_common_sequences(tokenized_messages)
        patterns['common_sequences'] = common_seqs
        
        # Extract positional patterns (tokens that appear at specific positions)
        positional = await self._extract_positional_patterns(tokenized_messages)
        patterns['positional_patterns'] = positional
        
        # Extract semantic structure
        semantic = await self._extract_semantic_structure(
            tokenized_messages, stats_result
        )
        patterns['semantic_structure'] = semantic
        
        return patterns

    async def _extract_message_templates(self, tokenized_messages: List[List[str]]) -> List[Dict[str, Any]]:
        """Extract common message structure templates."""
        if not tokenized_messages:
            return []
        
        # Group messages by token count
        by_length = defaultdict(list)
        for tokens in tokenized_messages:
            by_length[len(tokens)].append(tokens)
        
        templates = []
        
        for length, message_group in by_length.items():
            if len(message_group) < 2:  # Need at least 2 messages to find a template
                continue
            
            # Find common positions
            template = ["<VAR>"] * length
            position_stats = defaultdict(Counter)
            
            for tokens in message_group:
                for pos, token in enumerate(tokens):
                    position_stats[pos][token] += 1
            
            # Determine fixed vs variable positions
            for pos in range(length):
                token_counts = position_stats[pos]
                most_common_token, count = token_counts.most_common(1)[0]
                
                # If token appears in >80% of messages at this position, it's fixed
                if count >= len(message_group) * 0.8:
                    template[pos] = most_common_token
            
            # Calculate template significance
            fixed_positions = sum(1 for t in template if t != "<VAR>")
            significance = (fixed_positions / length) * len(message_group)
            
            if significance > 1.0:  # Template is significant
                templates.append({
                    'template': template,
                    'frequency': len(message_group),
                    'significance': significance,
                    'length': length
                })
        
        # Sort by significance
        templates.sort(key=lambda t: t['significance'], reverse=True)
        return templates[:10]  # Return top 10 templates

    async def _extract_common_sequences(self, tokenized_messages: List[List[str]]) -> Dict[Tuple[str, ...], int]:
        """Extract common token sequences across messages."""
        sequence_counts = defaultdict(int)
        
        for tokens in tokenized_messages:
            # Extract n-grams of different lengths
            for n in range(2, min(6, len(tokens) + 1)):
                for i in range(len(tokens) - n + 1):
                    sequence = tuple(tokens[i:i + n])
                    sequence_counts[sequence] += 1
        
        # Filter significant sequences
        min_frequency = max(2, len(tokenized_messages) * 0.1)
        significant_sequences = {
            seq: count for seq, count in sequence_counts.items()
            if count >= min_frequency
        }
        
        return significant_sequences

    async def _extract_positional_patterns(self, tokenized_messages: List[List[str]]) -> Dict[int, Dict[str, int]]:
        """Extract patterns based on token positions."""
        positional_patterns = defaultdict(Counter)
        
        for tokens in tokenized_messages:
            for pos, token in enumerate(tokens):
                positional_patterns[pos][token] += 1
        
        # Filter significant positional patterns
        filtered_patterns = {}
        for pos, token_counts in positional_patterns.items():
            # Only include positions where some token has significant frequency
            max_count = token_counts.most_common(1)[0][1] if token_counts else 0
            if max_count >= len(tokenized_messages) * 0.3:  # 30% threshold
                filtered_patterns[pos] = dict(token_counts)
        
        return filtered_patterns

    async def _extract_semantic_structure(
        self, 
        tokenized_messages: List[List[str]], 
        stats_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract semantic structure information."""
        structure = {
            'has_header': False,
            'has_footer': False,
            'header_length': 0,
            'footer_length': 0,
            'body_variable': True,
            'field_structure': []
        }
        
        if not tokenized_messages:
            return structure
        
        # Analyze structural features from stats
        structural_features = stats_result.get('structural_features', {})
        
        # Detect header
        if structural_features.get('header_length'):
            structure['has_header'] = True
            structure['header_length'] = structural_features['header_length']
        
        # Detect footer
        if structural_features.get('footer_length'):
            structure['has_footer'] = True
            structure['footer_length'] = structural_features['footer_length']
        
        # Analyze body variability
        body_patterns = []
        header_len = structure['header_length']
        footer_len = structure['footer_length']
        
        for tokens in tokenized_messages:
            if len(tokens) > header_len + footer_len:
                body = tokens[header_len:len(tokens) - footer_len] if footer_len else tokens[header_len:]
                body_patterns.append(tuple(body))
        
        unique_bodies = len(set(body_patterns))
        structure['body_variable'] = unique_bodies > len(tokenized_messages) * 0.5
        
        return structure

    async def _identify_symbols(
        self, 
        tokenized_messages: List[List[str]], 
        structural_patterns: Dict[str, Any], 
        stats_result: Dict[str, Any]
    ) -> Dict[str, Symbol]:
        """Identify terminal and non-terminal symbols with semantic annotations."""
        symbols = {}
        
        # Collect all tokens and their frequencies
        token_counts = Counter()
        for tokens in tokenized_messages:
            for token in tokens:
                token_counts[token] += 1
        
        # Create terminal symbols
        for token, frequency in token_counts.items():
            semantic_type = self._infer_semantic_type(token, frequency, tokenized_messages)
            
            symbol = Symbol(
                name=token,
                is_terminal=True,
                frequency=frequency,
                semantic_type=semantic_type
            )
            symbols[token] = symbol
        
        # Create non-terminal symbols based on patterns
        non_terminals = await self._create_non_terminal_symbols(
            tokenized_messages, structural_patterns, stats_result
        )
        
        symbols.update(non_terminals)
        
        # Add standard non-terminals
        standard_nts = [
            "<START>", "<MESSAGE>", "<HEADER>", "<BODY>", "<FOOTER>",
            "<FIELD>", "<DATA>", "<LENGTH>", "<DELIMITER>"
        ]
        
        for nt_name in standard_nts:
            if nt_name not in symbols:
                symbols[nt_name] = Symbol(
                    name=nt_name,
                    is_terminal=False,
                    frequency=len(tokenized_messages),
                    semantic_type=nt_name.lower().strip('<>')
                )
        
        return symbols

    def _infer_semantic_type(self, token: str, frequency: int, messages: List[List[str]]) -> str:
        """Infer semantic type of a terminal symbol."""
        if token.startswith('0x') and len(token) == 4:  # Single byte
            byte_val = int(token[2:], 16)
            if byte_val == 0:
                return 'null'
            elif 32 <= byte_val <= 126:
                return 'ascii'
            elif byte_val in [0x0A, 0x0D, 0x09]:
                return 'whitespace'
            else:
                return 'binary'
        
        elif token.startswith('<PATTERN_'):
            return 'pattern'
        elif token.startswith('<FIELD_'):
            return 'field'
        elif token.startswith('<LENGTH'):
            return 'length'
        elif token.startswith('<DATA_'):
            return 'data'
        elif token.startswith('<TEXT'):
            return 'text'
        elif 'ENTROPY' in token:
            return 'entropy_region'
        else:
            return 'unknown'

    async def _create_non_terminal_symbols(
        self, 
        tokenized_messages: List[List[str]], 
        structural_patterns: Dict[str, Any], 
        stats_result: Dict[str, Any]
    ) -> Dict[str, Symbol]:
        """Create non-terminal symbols based on structural patterns."""
        non_terminals = {}
        
        # Create symbols for common sequences
        common_sequences = structural_patterns.get('common_sequences', {})
        for sequence, frequency in common_sequences.items():
            if len(sequence) > 1:
                nt_name = f"<SEQ_{len(sequence)}_{hash(sequence) % 10000}>"
                non_terminals[nt_name] = Symbol(
                    name=nt_name,
                    is_terminal=False,
                    frequency=frequency,
                    semantic_type='sequence'
                )
        
        # Create symbols for message templates
        templates = structural_patterns.get('message_templates', [])
        for i, template_info in enumerate(templates):
            nt_name = f"<TEMPLATE_{i}>"
            non_terminals[nt_name] = Symbol(
                name=nt_name,
                is_terminal=False,
                frequency=template_info['frequency'],
                semantic_type='template'
            )
        
        return non_terminals

    async def _generate_initial_rules(
        self, 
        tokenized_messages: List[List[str]], 
        symbols: Dict[str, Symbol], 
        structural_patterns: Dict[str, Any]
    ) -> List[ProductionRule]:
        """Generate initial production rules from patterns and structure."""
        rules = []
        
        # Start rule
        start_symbol = symbols["<START>"]
        message_symbol = symbols["<MESSAGE>"]
        
        start_rule = ProductionRule(
            left_hand_side=start_symbol,
            right_hand_side=[message_symbol],
            probability=1.0,
            frequency=len(tokenized_messages),
            semantic_role='root'
        )
        rules.append(start_rule)
        
        # Structure-based rules
        structure_rules = await self._generate_structure_rules(
            tokenized_messages, symbols, structural_patterns
        )
        rules.extend(structure_rules)
        
        # Pattern-based rules
        pattern_rules = await self._generate_pattern_rules(
            tokenized_messages, symbols, structural_patterns
        )
        rules.extend(pattern_rules)
        
        # Terminal rules
        terminal_rules = await self._generate_terminal_rules(tokenized_messages, symbols)
        rules.extend(terminal_rules)
        
        return rules

    async def _generate_structure_rules(
        self, 
        tokenized_messages: List[List[str]], 
        symbols: Dict[str, Symbol], 
        structural_patterns: Dict[str, Any]
    ) -> List[ProductionRule]:
        """Generate rules based on message structure."""
        rules = []
        
        semantic_structure = structural_patterns.get('semantic_structure', {})
        message_symbol = symbols["<MESSAGE>"]
        
        if semantic_structure.get('has_header') and semantic_structure.get('has_footer'):
            # MESSAGE -> HEADER BODY FOOTER
            rule = ProductionRule(
                left_hand_side=message_symbol,
                right_hand_side=[symbols["<HEADER>"], symbols["<BODY>"], symbols["<FOOTER>"]],
                probability=0.7,
                frequency=int(len(tokenized_messages) * 0.7),
                semantic_role='structured_message'
            )
            rules.append(rule)
            
        elif semantic_structure.get('has_header'):
            # MESSAGE -> HEADER BODY
            rule = ProductionRule(
                left_hand_side=message_symbol,
                right_hand_side=[symbols["<HEADER>"], symbols["<BODY>"]],
                probability=0.8,
                frequency=int(len(tokenized_messages) * 0.8),
                semantic_role='header_body_message'
            )
            rules.append(rule)
            
        elif semantic_structure.get('has_footer'):
            # MESSAGE -> BODY FOOTER
            rule = ProductionRule(
                left_hand_side=message_symbol,
                right_hand_side=[symbols["<BODY>"], symbols["<FOOTER>"]],
                probability=0.8,
                frequency=int(len(tokenized_messages) * 0.8),
                semantic_role='body_footer_message'
            )
            rules.append(rule)
        else:
            # MESSAGE -> BODY
            rule = ProductionRule(
                left_hand_side=message_symbol,
                right_hand_side=[symbols["<BODY>"]],
                probability=0.9,
                frequency=int(len(tokenized_messages) * 0.9),
                semantic_role='simple_message'
            )
            rules.append(rule)
        
        return rules

    async def _generate_pattern_rules(
        self, 
        tokenized_messages: List[List[str]], 
        symbols: Dict[str, Symbol], 
        structural_patterns: Dict[str, Any]
    ) -> List[ProductionRule]:
        """Generate rules based on detected patterns."""
        rules = []
        
        # Rules for common sequences
        common_sequences = structural_patterns.get('common_sequences', {})
        for sequence, frequency in common_sequences.items():
            if len(sequence) > 1:
                nt_name = f"<SEQ_{len(sequence)}_{hash(sequence) % 10000}>"
                if nt_name in symbols:
                    lhs_symbol = symbols[nt_name]
                    rhs_symbols = [symbols[token] for token in sequence if token in symbols]
                    
                    if len(rhs_symbols) == len(sequence):
                        rule = ProductionRule(
                            left_hand_side=lhs_symbol,
                            right_hand_side=rhs_symbols,
                            probability=frequency / len(tokenized_messages),
                            frequency=frequency,
                            semantic_role='sequence'
                        )
                        rules.append(rule)
        
        return rules

    async def _generate_terminal_rules(
        self, 
        tokenized_messages: List[List[str]], 
        symbols: Dict[str, Symbol]
    ) -> List[ProductionRule]:
        """Generate rules that produce terminal symbols."""
        rules = []
        
        # Count how often each terminal appears in different contexts
        context_counts = defaultdict(Counter)
        
        # Simple context: which non-terminal could produce this terminal
        for tokens in tokenized_messages:
            for token in tokens:
                if token in symbols and symbols[token].is_terminal:
                    # For now, assume all terminals can be produced by FIELD or DATA
                    context_counts['<FIELD>'][token] += 1
                    context_counts['<DATA>'][token] += 1
        
        # Generate rules
        for context, token_counts in context_counts.items():
            if context in symbols:
                lhs_symbol = symbols[context]
                total_count = sum(token_counts.values())
                
                for token, count in token_counts.items():
                    if token in symbols:
                        rule = ProductionRule(
                            left_hand_side=lhs_symbol,
                            right_hand_side=[symbols[token]],
                            probability=count / total_count,
                            frequency=count,
                            semantic_role='terminal'
                        )
                        rules.append(rule)
        
        return rules

    async def _refine_grammar(
        self, 
        initial_rules: List[ProductionRule], 
        tokenized_messages: List[List[str]], 
        symbols: Dict[str, Symbol], 
        stats_result: Dict[str, Any]
    ) -> List[ProductionRule]:
        """Refine grammar using EM algorithm and semantic analysis."""
        self.logger.debug("Refining grammar with EM algorithm")
        
        current_rules = initial_rules.copy()
        
        for iteration in range(self.max_em_iterations):
            self.logger.debug(f"EM iteration {iteration + 1}/{self.max_em_iterations}")
            
            # E-step: Calculate expected counts
            expected_counts = await self._calculate_expected_counts(
                current_rules, tokenized_messages
            )
            
            # M-step: Update probabilities
            new_rules = await self._update_rule_probabilities(
                current_rules, expected_counts
            )
            
            # Check convergence
            if await self._has_converged(current_rules, new_rules):
                self.logger.debug(f"EM converged after {iteration + 1} iterations")
                break
            
            current_rules = new_rules
        
        # Additional refinement with semantic analysis
        if self.semantic_analysis_enabled:
            current_rules = await self._semantic_refinement(
                current_rules, tokenized_messages, stats_result
            )
        
        return current_rules

    async def _calculate_expected_counts(
        self, 
        rules: List[ProductionRule], 
        tokenized_messages: List[List[str]]
    ) -> Dict[str, float]:
        """Calculate expected rule counts (E-step of EM)."""
        expected_counts = defaultdict(float)
        
        for tokens in tokenized_messages:
            rule_usage = await self._count_rule_usage_in_message(rules, tokens)
            for rule_str, count in rule_usage.items():
                expected_counts[rule_str] += count
        
        return dict(expected_counts)

    async def _count_rule_usage_in_message(
        self, 
        rules: List[ProductionRule], 
        tokens: List[str]
    ) -> Dict[str, float]:
        """Count how many times each rule is used to generate a message."""
        rule_usage = defaultdict(float)
        
        # Simplified counting based on rule matching
        for rule in rules:
            rhs_names = [s.name for s in rule.right_hand_side]
            
            if len(rhs_names) == 1:
                # Terminal rule
                count = tokens.count(rhs_names[0])
                rule_usage[str(rule)] += count
            else:
                # Multi-token rule - count subsequence occurrences
                count = self._count_subsequence_in_tokens(tokens, rhs_names)
                rule_usage[str(rule)] += count
        
        return dict(rule_usage)

    def _count_subsequence_in_tokens(self, tokens: List[str], pattern: List[str]) -> int:
        """Count occurrences of pattern in token sequence."""
        count = 0
        for i in range(len(tokens) - len(pattern) + 1):
            if tokens[i:i + len(pattern)] == pattern:
                count += 1
        return count

    async def _update_rule_probabilities(
        self, 
        rules: List[ProductionRule], 
        expected_counts: Dict[str, float]
    ) -> List[ProductionRule]:
        """Update rule probabilities (M-step of EM)."""
        # Group rules by left-hand side
        rules_by_lhs = defaultdict(list)
        for rule in rules:
            rules_by_lhs[rule.left_hand_side.name].append(rule)
        
        updated_rules = []
        
        for lhs_name, lhs_rules in rules_by_lhs.items():
            # Calculate total expected count for this LHS
            total_count = sum(expected_counts.get(str(rule), 0) for rule in lhs_rules)
            
            if total_count > 0:
                for rule in lhs_rules:
                    expected_count = expected_counts.get(str(rule), 0)
                    new_probability = expected_count / total_count
                    
                    # Create updated rule
                    updated_rule = ProductionRule(
                        left_hand_side=rule.left_hand_side,
                        right_hand_side=rule.right_hand_side,
                        probability=new_probability,
                        frequency=int(expected_count),
                        contexts=rule.contexts,
                        semantic_role=rule.semantic_role,
                        confidence=rule.confidence
                    )
                    updated_rules.append(updated_rule)
            else:
                # No counts - keep original rules
                updated_rules.extend(lhs_rules)
        
        return updated_rules

    async def _has_converged(
        self, 
        old_rules: List[ProductionRule], 
        new_rules: List[ProductionRule]
    ) -> bool:
        """Check if EM algorithm has converged."""
        if len(old_rules) != len(new_rules):
            return False
        
        # Create probability mappings
        old_probs = {str(rule): rule.probability for rule in old_rules}
        new_probs = {str(rule): rule.probability for rule in new_rules}
        
        # Check maximum probability change
        max_change = 0.0
        for rule_str in old_probs:
            if rule_str in new_probs:
                change = abs(old_probs[rule_str] - new_probs[rule_str])
                max_change = max(max_change, change)
        
        return max_change < self.convergence_threshold

    async def _semantic_refinement(
        self, 
        rules: List[ProductionRule], 
        tokenized_messages: List[List[str]], 
        stats_result: Dict[str, Any]
    ) -> List[ProductionRule]:
        """Apply semantic analysis to refine grammar rules."""
        refined_rules = []
        
        for rule in rules:
            # Calculate confidence based on semantic consistency
            confidence = await self._calculate_semantic_confidence(
                rule, tokenized_messages, stats_result
            )
            
            # Update rule with confidence
            refined_rule = ProductionRule(
                left_hand_side=rule.left_hand_side,
                right_hand_side=rule.right_hand_side,
                probability=rule.probability,
                frequency=rule.frequency,
                contexts=rule.contexts,
                semantic_role=rule.semantic_role,
                confidence=confidence
            )
            
            # Only keep rules with sufficient confidence
            if confidence >= 0.1:  # Minimum confidence threshold
                refined_rules.append(refined_rule)
        
        return refined_rules

    async def _calculate_semantic_confidence(
        self, 
        rule: ProductionRule, 
        tokenized_messages: List[List[str]], 
        stats_result: Dict[str, Any]
    ) -> float:
        """Calculate semantic confidence for a production rule."""
        base_confidence = rule.frequency / len(tokenized_messages)
        
        # Adjust based on rule type
        if rule.is_terminal_rule():
            # Terminal rules get bonus for being simple
            confidence = min(1.0, base_confidence * 1.2)
        elif rule.is_recursive():
            # Recursive rules get penalty for complexity
            confidence = base_confidence * 0.8
        else:
            confidence = base_confidence
        
        # Adjust based on semantic role
        if rule.semantic_role in ['root', 'structured_message', 'header_body_message']:
            confidence *= 1.1  # Boost structural rules
        elif rule.semantic_role == 'terminal':
            confidence *= 0.9  # Slight penalty for generic terminal rules
        
        return min(1.0, max(0.0, confidence))

    async def _optimize_grammar(
        self, 
        rules: List[ProductionRule], 
        symbols: Dict[str, Symbol]
    ) -> List[ProductionRule]:
        """Optimize grammar by removing redundant rules and merging similar ones."""
        optimized_rules = []
        
        # Remove very low probability rules
        for rule in rules:
            if rule.probability >= 0.01 or rule.frequency >= 2:
                optimized_rules.append(rule)
        
        # TODO: Implement rule merging and redundancy removal
        # This is a complex task that would involve:
        # 1. Identifying semantically equivalent rules
        # 2. Merging rules with similar right-hand sides
        # 3. Eliminating unreachable non-terminals
        # 4. Factoring out common prefixes/suffixes
        
        return optimized_rules

    def _generate_cache_key(self, messages: List[bytes], protocol_hint: Optional[str]) -> str:
        """Generate cache key for memoization."""
        # Use hash of first few messages and protocol hint
        content = b''.join(messages[:5])  # First 5 messages
        hint_str = protocol_hint or ""
        combined = content + hint_str.encode('utf-8')
        return hashlib.sha256(combined).hexdigest()[:16]

    async def save_grammar(self, grammar: Grammar, filepath: str) -> None:
        """Save grammar to file."""
        try:
            grammar_dict = grammar.to_dict()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(grammar_dict, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Grammar saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save grammar: {e}")
            raise ModelException(f"Grammar save error: {e}")

    async def load_grammar(self, filepath: str) -> Grammar:
        """Load grammar from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                grammar_dict = json.load(f)
            
            # Reconstruct symbols
            symbols = {}
            for name, symbol_data in grammar_dict['symbols'].items():
                symbols[name] = Symbol(
                    name=symbol_data['name'],
                    is_terminal=symbol_data['is_terminal'],
                    frequency=symbol_data.get('frequency', 0),
                    entropy=symbol_data.get('entropy', 0.0),
                    semantic_type=symbol_data.get('semantic_type')
                )
            
            # Reconstruct rules
            rules = []
            for rule_data in grammar_dict['rules']:
                lhs = symbols[rule_data['lhs']]
                rhs = [symbols[name] for name in rule_data['rhs']]
                
                rule = ProductionRule(
                    left_hand_side=lhs,
                    right_hand_side=rhs,
                    probability=rule_data['probability'],
                    frequency=rule_data.get('frequency', 0),
                    semantic_role=rule_data.get('semantic_role'),
                    confidence=rule_data.get('confidence', 0.0)
                )
                rules.append(rule)
            
            grammar = Grammar(
                rules=rules,
                symbols=symbols,
                start_symbol=grammar_dict.get('start_symbol', '<START>'),
                metadata=grammar_dict.get('metadata', {})
            )
            
            self.logger.info(f"Grammar loaded from {filepath}")
            return grammar
            
        except Exception as e:
            self.logger.error(f"Failed to load grammar: {e}")
            raise ModelException(f"Grammar load error: {e}")

    async def shutdown(self):
        """Shutdown the grammar learner and cleanup resources."""
        self.logger.info("Shutting down Grammar Learner")
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.statistical_analyzer:
            await self.statistical_analyzer.shutdown()
        
        # Clear caches
        self._message_cache.clear()
        self._grammar_cache.clear()
        self._symbol_registry.clear()
        
        self.logger.info("Grammar Learner shutdown completed")