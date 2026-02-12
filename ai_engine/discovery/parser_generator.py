"""
QBITEL Engine - Dynamic Parser Generator

This module implements dynamic parser generation from learned grammars, including
code generation for runtime parsers, validation logic, and performance optimization.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
import inspect
import ast
import types
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
import struct

from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException
from .grammar_learner import Grammar, ProductionRule, Symbol


@dataclass
class ParseResult:
    """Result of parsing operation."""

    success: bool
    parsed_data: Dict[str, Any]
    remaining_data: bytes
    parse_tree: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParserNode:
    """Node in the parse tree."""

    symbol_name: str
    value: Optional[Union[bytes, str]] = None
    children: List["ParserNode"] = field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 1.0
    semantic_type: Optional[str] = None


@dataclass
class GeneratedParser:
    """Container for a dynamically generated parser."""

    parser_id: str
    grammar: Grammar
    parse_function: Callable[[bytes], ParseResult]
    validate_function: Callable[[bytes], bool]
    metadata: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class ParserTemplate:
    """Template for generating parser code."""

    PARSER_TEMPLATE = '''
async def parse_{parser_id}(data: bytes) -> ParseResult:
    """Generated parser for protocol {protocol_name}."""
    parser = ProtocolParser_{parser_id}(data)
    return await parser.parse()

class ProtocolParser_{parser_id}:
    """Generated parser class for {protocol_name}."""
    
    def __init__(self, data: bytes):
        self.data = data
        self.position = 0
        self.parse_stack = []
        self.errors = []
        self.confidence_scores = []
        self.memoization_cache = {{}} if {enable_memoization} else None
        
    async def parse(self) -> ParseResult:
        """Main parsing method."""
        try:
            root_node = await self.parse_start()
            
            if root_node and self.position <= len(self.data):
                remaining = self.data[self.position:]
                confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0
                
                return ParseResult(
                    success=True,
                    parsed_data=self._node_to_dict(root_node),
                    remaining_data=remaining,
                    parse_tree=self._node_to_tree(root_node),
                    confidence=confidence,
                    errors=self.errors
                )
            else:
                return ParseResult(
                    success=False,
                    parsed_data={{}},
                    remaining_data=self.data,
                    confidence=0.0,
                    errors=self.errors or ["Failed to parse start symbol"]
                )
                
        except Exception as e:
            return ParseResult(
                success=False,
                parsed_data={{}},
                remaining_data=self.data,
                confidence=0.0,
                errors=[str(e)]
            )
    
{parsing_methods}
    
    def _node_to_dict(self, node: ParserNode) -> Dict[str, Any]:
        """Convert parse tree node to dictionary."""
        if not node.children:
            return {{
                "type": node.symbol_name,
                "value": node.value,
                "position": [node.start_pos, node.end_pos],
                "confidence": node.confidence,
                "semantic_type": node.semantic_type
            }}
        
        return {{
            "type": node.symbol_name,
            "children": [self._node_to_dict(child) for child in node.children],
            "position": [node.start_pos, node.end_pos],
            "confidence": node.confidence,
            "semantic_type": node.semantic_type
        }}
    
    def _node_to_tree(self, node: ParserNode) -> Dict[str, Any]:
        """Convert parse tree node to simplified tree structure."""
        if not node.children:
            return {{"symbol": node.symbol_name, "value": node.value}}
        
        return {{
            "symbol": node.symbol_name,
            "children": [self._node_to_tree(child) for child in node.children]
        }}
    
    def _read_bytes(self, count: int) -> Optional[bytes]:
        """Read specified number of bytes from current position."""
        if self.position + count <= len(self.data):
            result = self.data[self.position:self.position + count]
            self.position += count
            return result
        return None
    
    def _peek_bytes(self, count: int) -> Optional[bytes]:
        """Peek at bytes without advancing position."""
        if self.position + count <= len(self.data):
            return self.data[self.position:self.position + count]
        return None
    
    def _match_literal(self, literal: bytes) -> bool:
        """Match literal byte sequence."""
        if self._peek_bytes(len(literal)) == literal:
            self.position += len(literal)
            return True
        return False
    
    def _record_confidence(self, score: float):
        """Record confidence score for current parsing step."""
        self.confidence_scores.append(max(0.0, min(1.0, score)))
    
    def _get_memo_key(self, symbol: str, position: int) -> str:
        """Get memoization key."""
        return f"{{symbol}}:{{position}}"
    
    def _memo_get(self, symbol: str, position: int) -> Optional[Any]:
        """Get memoized result."""
        if self.memoization_cache is not None:
            return self.memoization_cache.get(self._get_memo_key(symbol, position))
        return None
    
    def _memo_set(self, symbol: str, position: int, result: Any):
        """Set memoized result."""
        if self.memoization_cache is not None:
            self.memoization_cache[self._get_memo_key(symbol, position)] = result
'''


class ParserGenerator:
    """
    Dynamic parser generator for protocol discovery.

    This class generates efficient parsers from learned grammars, with support
    for backtracking, error recovery, and performance optimization.
    """

    def __init__(self, config: Config):
        """Initialize the parser generator."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Generation parameters
        self.enable_backtracking = True
        self.enable_error_recovery = True
        self.max_recursion_depth = 20
        self.enable_memoization = True
        self.optimize_generated_code = True

        # Performance settings
        self.use_parallel_generation = True
        self.max_workers = config.inference.num_workers if hasattr(config, "inference") else 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Parser cache and registry
        self._parser_cache: Dict[str, GeneratedParser] = {}
        self._code_cache: Dict[str, str] = {}
        self._compiled_parsers: Dict[str, types.ModuleType] = {}

        self.logger.info("Parser Generator initialized")

    async def generate_parser(
        self,
        grammar: Grammar,
        parser_id: Optional[str] = None,
        protocol_name: Optional[str] = None,
    ) -> GeneratedParser:
        """
        Generate a dynamic parser from a learned grammar.

        Args:
            grammar: Learned grammar to generate parser from
            parser_id: Optional unique identifier for parser
            protocol_name: Optional protocol name for documentation

        Returns:
            Generated parser with parse and validation functions
        """
        if not grammar.rules:
            raise ProtocolException("Cannot generate parser from empty grammar")

        start_time = time.time()

        # Generate parser ID if not provided
        if not parser_id:
            parser_id = self._generate_parser_id(grammar)

        # Check cache
        if parser_id in self._parser_cache:
            self.logger.info(f"Returning cached parser {parser_id}")
            return self._parser_cache[parser_id]

        self.logger.info(f"Generating parser {parser_id} from grammar with {len(grammar.rules)} rules")

        try:
            # Validate grammar
            await self._validate_grammar(grammar)

            # Analyze grammar for optimization
            grammar_analysis = await self._analyze_grammar(grammar)

            # Generate parser code
            parser_code = await self._generate_parser_code(grammar, parser_id, protocol_name or "unknown", grammar_analysis)

            # Compile parser
            compiled_parser = await self._compile_parser(parser_code, parser_id)

            # Create parser functions
            parse_function = self._create_parse_function(compiled_parser, parser_id)
            validate_function = self._create_validate_function(compiled_parser, parser_id)

            # Create generated parser object
            generated_parser = GeneratedParser(
                parser_id=parser_id,
                grammar=grammar,
                parse_function=parse_function,
                validate_function=validate_function,
                metadata={
                    "protocol_name": protocol_name,
                    "generation_time": time.time() - start_time,
                    "rule_count": len(grammar.rules),
                    "symbol_count": len(grammar.symbols),
                    "analysis": grammar_analysis,
                    "optimizations_enabled": self.optimize_generated_code,
                },
            )

            # Cache result
            self._parser_cache[parser_id] = generated_parser

            self.logger.info(f"Parser generation completed in {time.time() - start_time:.2f}s")
            return generated_parser

        except Exception as e:
            self.logger.error(f"Parser generation failed: {e}")
            raise ModelException(f"Parser generation error: {e}")

    async def _validate_grammar(self, grammar: Grammar) -> None:
        """Validate that grammar is suitable for parser generation."""
        # Check for start symbol
        if grammar.start_symbol not in grammar.symbols:
            raise ProtocolException(f"Start symbol {grammar.start_symbol} not found in grammar")

        # Check for unreachable symbols
        reachable_symbols = await self._find_reachable_symbols(grammar)
        unreachable = set(grammar.symbols.keys()) - reachable_symbols

        if unreachable:
            self.logger.warning(f"Unreachable symbols found: {unreachable}")

        # Check for left recursion
        left_recursive = await self._detect_left_recursion(grammar)
        if left_recursive:
            self.logger.warning(f"Left recursive rules detected: {left_recursive}")

    async def _find_reachable_symbols(self, grammar: Grammar) -> Set[str]:
        """Find all symbols reachable from start symbol."""
        reachable = set()
        to_visit = deque([grammar.start_symbol])

        while to_visit:
            symbol = to_visit.popleft()
            if symbol in reachable:
                continue

            reachable.add(symbol)

            # Add symbols from RHS of rules for this symbol
            for rule in grammar.get_rules_for_symbol(symbol):
                for rhs_symbol in rule.right_hand_side:
                    if rhs_symbol.name not in reachable:
                        to_visit.append(rhs_symbol.name)

        return reachable

    async def _detect_left_recursion(self, grammar: Grammar) -> List[str]:
        """Detect left recursive rules."""
        left_recursive = []

        for symbol_name in grammar.symbols:
            if await self._is_left_recursive(grammar, symbol_name, set()):
                left_recursive.append(symbol_name)

        return left_recursive

    async def _is_left_recursive(self, grammar: Grammar, symbol: str, visited: Set[str]) -> bool:
        """Check if symbol is left recursive."""
        if symbol in visited:
            return True

        visited.add(symbol)

        for rule in grammar.get_rules_for_symbol(symbol):
            if rule.right_hand_side and not rule.right_hand_side[0].is_terminal:
                first_symbol = rule.right_hand_side[0].name
                if first_symbol == symbol:
                    return True
                elif await self._is_left_recursive(grammar, first_symbol, visited.copy()):
                    return True

        return False

    async def _analyze_grammar(self, grammar: Grammar) -> Dict[str, Any]:
        """Analyze grammar for optimization opportunities."""
        analysis = {
            "terminal_count": len(grammar.get_terminal_symbols()),
            "non_terminal_count": len(grammar.get_non_terminal_symbols()),
            "rule_count": len(grammar.rules),
            "recursive_rules": [],
            "terminal_rules": [],
            "complex_rules": [],
            "optimization_opportunities": [],
        }

        # Analyze rules
        for rule in grammar.rules:
            if rule.is_recursive():
                analysis["recursive_rules"].append(rule.left_hand_side.name)

            if rule.is_terminal_rule():
                analysis["terminal_rules"].append(rule.left_hand_side.name)

            if len(rule.right_hand_side) > 5:
                analysis["complex_rules"].append(rule.left_hand_side.name)

        # Identify optimization opportunities
        if len(analysis["terminal_rules"]) > 20:
            analysis["optimization_opportunities"].append("terminal_rule_consolidation")

        if len(analysis["recursive_rules"]) > 5:
            analysis["optimization_opportunities"].append("recursion_optimization")

        return analysis

    async def _generate_parser_code(
        self,
        grammar: Grammar,
        parser_id: str,
        protocol_name: str,
        analysis: Dict[str, Any],
    ) -> str:
        """Generate Python parser code from grammar."""
        self.logger.debug(f"Generating parser code for {parser_id}")

        # Generate parsing methods for each non-terminal
        parsing_methods = []

        for symbol_name, symbol in grammar.symbols.items():
            if not symbol.is_terminal:
                method_code = await self._generate_parsing_method(grammar, symbol_name)
                parsing_methods.append(method_code)

        # Combine all parsing methods
        all_methods = "\n".join(parsing_methods)

        # Generate complete parser code
        parser_code = ParserTemplate.PARSER_TEMPLATE.format(
            parser_id=parser_id,
            protocol_name=protocol_name,
            enable_memoization=str(self.enable_memoization).lower(),
            parsing_methods=all_methods,
        )

        return parser_code

    async def _generate_parsing_method(self, grammar: Grammar, symbol_name: str) -> str:
        """Generate parsing method for a specific symbol."""
        rules = grammar.get_rules_for_symbol(symbol_name)
        if not rules:
            return f"""
    async def parse_{self._sanitize_symbol_name(symbol_name)}(self) -> Optional[ParserNode]:
        \"\"\"Parse {symbol_name} symbol.\"\"\"
        self.errors.append(f"No rules found for {symbol_name}")
        return None
"""

        # Generate parsing logic based on rules
        parsing_logic = await self._generate_rule_parsing_logic(rules, symbol_name)

        method_code = f"""
    async def parse_{self._sanitize_symbol_name(symbol_name)}(self) -> Optional[ParserNode]:
        \"\"\"Parse {symbol_name} symbol.\"\"\"
        start_pos = self.position
        
        # Check memoization
        if self.memoization_cache is not None:
            memo_result = self._memo_get("{symbol_name}", start_pos)
            if memo_result is not None:
                if memo_result:
                    self.position = memo_result.end_pos
                return memo_result
        
{parsing_logic}
        
        # Memoize failure
        if self.memoization_cache is not None:
            self._memo_set("{symbol_name}", start_pos, None)
        
        return None  # Failed to parse
"""

        return method_code

    async def _generate_rule_parsing_logic(self, rules: List[ProductionRule], symbol_name: str) -> str:
        """Generate parsing logic for a set of rules."""
        if len(rules) == 1:
            # Single rule - no alternatives
            return await self._generate_single_rule_logic(rules[0], symbol_name)
        else:
            # Multiple rules - try alternatives
            return await self._generate_alternative_rules_logic(rules, symbol_name)

    async def _generate_single_rule_logic(self, rule: ProductionRule, symbol_name: str) -> str:
        """Generate logic for parsing a single rule."""
        if rule.is_terminal_rule():
            # Terminal rule - match literal
            terminal = rule.right_hand_side[0]
            return f"""
        # Single terminal rule: {rule}
        if self._match_terminal("{terminal.name}"):
            node = ParserNode(
                symbol_name="{symbol_name}",
                value=self.data[start_pos:self.position],
                start_pos=start_pos,
                end_pos=self.position,
                confidence={rule.probability},
                semantic_type="{terminal.semantic_type}"
            )
            self._record_confidence({rule.probability})
            
            # Memoize success
            if self.memoization_cache is not None:
                self._memo_set("{symbol_name}", start_pos, node)
            
            return node
"""
        else:
            # Non-terminal rule - parse sequence
            return await self._generate_sequence_parsing_logic(rule, symbol_name)

    async def _generate_sequence_parsing_logic(self, rule: ProductionRule, symbol_name: str) -> str:
        """Generate logic for parsing a sequence of symbols."""
        parsing_steps = []

        for i, symbol in enumerate(rule.right_hand_side):
            var_name = f"child_{i}"

            if symbol.is_terminal:
                parsing_steps.append(f"""
        # Parse terminal: {symbol.name}
        if not self._match_terminal("{symbol.name}"):
            self.position = start_pos  # Backtrack
            return None
        
        {var_name} = ParserNode(
            symbol_name="{symbol.name}",
            value=self.data[self.position - len("{symbol.name}"):self.position],
            start_pos=self.position - len("{symbol.name}"),
            end_pos=self.position,
            confidence=1.0,
            semantic_type="{symbol.semantic_type}"
        )
""")
            else:
                method_name = self._sanitize_symbol_name(symbol.name)
                parsing_steps.append(f"""
        # Parse non-terminal: {symbol.name}
        {var_name} = await self.parse_{method_name}()
        if {var_name} is None:
            self.position = start_pos  # Backtrack
            return None
""")

        # Build result node
        children_list = ", ".join([f"child_{i}" for i in range(len(rule.right_hand_side))])

        result_logic = f"""
        # Build result node
        children = [{children_list}]
        node = ParserNode(
            symbol_name="{symbol_name}",
            children=children,
            start_pos=start_pos,
            end_pos=self.position,
            confidence={rule.probability},
            semantic_type="{rule.semantic_role}"
        )
        self._record_confidence({rule.probability})
        
        # Memoize success
        if self.memoization_cache is not None:
            self._memo_set("{symbol_name}", start_pos, node)
        
        return node
"""

        return "\n".join(parsing_steps) + result_logic

    async def _generate_alternative_rules_logic(self, rules: List[ProductionRule], symbol_name: str) -> str:
        """Generate logic for trying alternative rules."""
        alternatives = []

        # Sort rules by probability (try most likely first)
        sorted_rules = sorted(rules, key=lambda r: r.probability, reverse=True)

        for rule in sorted_rules:
            if rule.is_terminal_rule():
                terminal = rule.right_hand_side[0]
                alt_logic = f"""
        # Try alternative: {rule}
        save_pos = self.position
        if self._match_terminal("{terminal.name}"):
            node = ParserNode(
                symbol_name="{symbol_name}",
                value=self.data[start_pos:self.position],
                start_pos=start_pos,
                end_pos=self.position,
                confidence={rule.probability},
                semantic_type="{terminal.semantic_type}"
            )
            self._record_confidence({rule.probability})
            
            # Memoize success
            if self.memoization_cache is not None:
                self._memo_set("{symbol_name}", start_pos, node)
            
            return node
        self.position = save_pos  # Restore position
"""
            else:
                # Generate sequence parsing for this alternative
                sequence_logic = await self._generate_sequence_parsing_logic(rule, symbol_name)
                alt_logic = f"""
        # Try alternative: {rule}
        save_pos = self.position
        try:
{self._indent_code(sequence_logic, 12)}
        except:
            self.position = save_pos  # Restore position
"""

            alternatives.append(alt_logic)

        return "\n".join(alternatives)

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = " " * spaces
        lines = code.strip().split("\n")
        return "\n".join([indent + line if line.strip() else line for line in lines])

    def _sanitize_symbol_name(self, symbol_name: str) -> str:
        """Sanitize symbol name for use as Python method name."""
        sanitized = symbol_name.replace("<", "").replace(">", "").replace("-", "_")
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
        return sanitized.lower()

    async def _compile_parser(self, parser_code: str, parser_id: str) -> types.ModuleType:
        """Compile generated parser code into executable module."""
        self.logger.debug(f"Compiling parser code for {parser_id}")

        try:
            # Create module
            module_name = f"generated_parser_{parser_id}"
            module = types.ModuleType(module_name)

            # Add required imports to module
            exec_globals = {
                "asyncio": asyncio,
                "Optional": Optional,
                "Dict": Dict,
                "Any": Any,
                "List": List,
                "ParserNode": ParserNode,
                "ParseResult": ParseResult,
            }

            # Add parser template helper methods
            helper_methods = self._generate_helper_methods()
            full_code = helper_methods + "\n" + parser_code

            # Compile and execute code
            compiled_code = compile(full_code, f"<generated_{parser_id}>", "exec")
            exec(compiled_code, exec_globals, module.__dict__)

            # Cache compiled parser
            self._compiled_parsers[parser_id] = module
            self._code_cache[parser_id] = full_code

            return module

        except Exception as e:
            self.logger.error(f"Failed to compile parser {parser_id}: {e}")
            self.logger.debug(f"Generated code:\n{parser_code}")
            raise ModelException(f"Parser compilation error: {e}")

    def _generate_helper_methods(self) -> str:
        """Generate helper methods for parser."""
        return '''
# Helper methods for generated parsers

def _match_terminal_helper(self, terminal_name: str) -> bool:
    """Helper to match terminal symbols."""
    if terminal_name.startswith("0x") and len(terminal_name) == 4:
        # Single byte literal
        try:
            byte_val = int(terminal_name[2:], 16)
            expected = bytes([byte_val])
            return self._match_literal(expected)
        except ValueError:
            return False
    elif terminal_name.startswith("<") and terminal_name.endswith(">"):
        # Complex terminal - simplified matching
        return True  # For now, accept any complex terminal
    else:
        # String literal
        try:
            expected = terminal_name.encode('utf-8')
            return self._match_literal(expected)
        except UnicodeEncodeError:
            return False

# Monkey patch the helper into parser classes
def _patch_parser_class(cls):
    """Patch parser class with helper methods."""
    cls._match_terminal = _match_terminal_helper
    return cls
'''

    def _create_parse_function(self, compiled_parser: types.ModuleType, parser_id: str) -> Callable[[bytes], ParseResult]:
        """Create parse function from compiled parser."""
        parse_func_name = f"parse_{parser_id}"

        if hasattr(compiled_parser, parse_func_name):
            return getattr(compiled_parser, parse_func_name)
        else:
            # Fallback function
            async def fallback_parse(data: bytes) -> ParseResult:
                return ParseResult(
                    success=False,
                    parsed_data={},
                    remaining_data=data,
                    confidence=0.0,
                    errors=[f"Parser function {parse_func_name} not found"],
                )

            return fallback_parse

    def _create_validate_function(self, compiled_parser: types.ModuleType, parser_id: str) -> Callable[[bytes], bool]:
        """Create validation function from compiled parser."""
        parse_func_name = f"parse_{parser_id}"

        if hasattr(compiled_parser, parse_func_name):
            parse_func = getattr(compiled_parser, parse_func_name)

            async def validate(data: bytes) -> bool:
                try:
                    result = await parse_func(data)
                    return result.success and result.confidence > 0.5
                except Exception:
                    return False

            return validate
        else:
            # Fallback validation
            async def fallback_validate(data: bytes) -> bool:
                return False

            return fallback_validate

    def _generate_parser_id(self, grammar: Grammar) -> str:
        """Generate unique parser ID from grammar."""
        # Create hash from grammar rules
        rule_strings = [str(rule) for rule in grammar.rules]
        combined = "\n".join(sorted(rule_strings))
        hash_obj = hashlib.sha256(combined.encode("utf-8"))
        return f"parser_{hash_obj.hexdigest()[:12]}"

    async def get_parser_code(self, parser_id: str) -> Optional[str]:
        """Get generated code for a parser."""
        return self._code_cache.get(parser_id)

    async def list_parsers(self) -> List[Dict[str, Any]]:
        """List all generated parsers."""
        parser_list = []

        for parser_id, parser in self._parser_cache.items():
            parser_info = {
                "parser_id": parser_id,
                "protocol_name": parser.metadata.get("protocol_name", "unknown"),
                "created_at": parser.created_at,
                "rule_count": parser.metadata.get("rule_count", 0),
                "symbol_count": parser.metadata.get("symbol_count", 0),
                "generation_time": parser.metadata.get("generation_time", 0.0),
            }
            parser_list.append(parser_info)

        return parser_list

    async def remove_parser(self, parser_id: str) -> bool:
        """Remove a generated parser from cache."""
        removed = False

        if parser_id in self._parser_cache:
            del self._parser_cache[parser_id]
            removed = True

        if parser_id in self._code_cache:
            del self._code_cache[parser_id]

        if parser_id in self._compiled_parsers:
            del self._compiled_parsers[parser_id]

        if removed:
            self.logger.info(f"Removed parser {parser_id}")

        return removed

    async def benchmark_parser(self, parser: GeneratedParser, test_data: List[bytes], iterations: int = 100) -> Dict[str, Any]:
        """Benchmark parser performance."""
        self.logger.info(f"Benchmarking parser {parser.parser_id} with {len(test_data)} samples")

        total_time = 0.0
        successful_parses = 0
        total_confidence = 0.0

        for _ in range(iterations):
            for data in test_data:
                start_time = time.time()

                try:
                    result = await parser.parse_function(data)
                    parse_time = time.time() - start_time
                    total_time += parse_time

                    if result.success:
                        successful_parses += 1
                        total_confidence += result.confidence

                except Exception as e:
                    self.logger.debug(f"Parse error during benchmark: {e}")
                    total_time += time.time() - start_time

        total_operations = iterations * len(test_data)

        return {
            "parser_id": parser.parser_id,
            "total_operations": total_operations,
            "total_time": total_time,
            "average_time": (total_time / total_operations if total_operations > 0 else 0.0),
            "operations_per_second": (total_operations / total_time if total_time > 0 else 0.0),
            "success_rate": (successful_parses / total_operations if total_operations > 0 else 0.0),
            "average_confidence": (total_confidence / successful_parses if successful_parses > 0 else 0.0),
            "successful_parses": successful_parses,
        }

    async def shutdown(self):
        """Shutdown parser generator and cleanup resources."""
        self.logger.info("Shutting down Parser Generator")

        if self.executor:
            self.executor.shutdown(wait=True)

        # Clear caches
        self._parser_cache.clear()
        self._code_cache.clear()
        self._compiled_parsers.clear()

        self.logger.info("Parser Generator shutdown completed")
