"""
CRONOS AI Engine - Message Validator

This module implements comprehensive message validation including syntax checking,
semantic analysis, integrity verification, and protocol compliance validation.
"""

import asyncio
import logging
import time
import struct
import zlib
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable, AsyncIterator
from enum import Enum
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor

from ..core.config import Config
from ..core.exceptions import ProtocolException, ModelException
from .grammar_learner import Grammar, ProductionRule, Symbol
from .parser_generator import ParseResult, GeneratedParser


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"              # Basic syntax and structure
    STANDARD = "standard"        # Standard compliance checks
    STRICT = "strict"           # Comprehensive validation
    ENTERPRISE = "enterprise"    # Enterprise-grade validation with custom rules


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a message."""
    severity: ValidationSeverity
    code: str
    message: str
    position: Optional[int] = None
    field_name: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of message validation."""
    is_valid: bool
    confidence: float
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_errors(self) -> bool:
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues 
                  if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
    
    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)


@dataclass
class ValidationRule:
    """Represents a validation rule."""
    name: str
    description: str
    validator_func: Callable[[bytes, Dict[str, Any]], List[ValidationIssue]]
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True
    protocol_specific: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageValidator:
    """
    Enterprise-grade message validator for protocol compliance.
    
    This class provides comprehensive validation capabilities including
    syntax checking, semantic analysis, integrity verification, and
    custom rule-based validation.
    """
    
    def __init__(self, config: Config):
        """Initialize the message validator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validation configuration
        self.default_validation_level = ValidationLevel.STANDARD
        self.enable_semantic_validation = True
        self.enable_integrity_checks = True
        self.enable_performance_validation = True
        
        # Performance settings
        self.max_message_size = 10 * 1024 * 1024  # 10MB
        self.validation_timeout = 30.0  # seconds
        self.use_parallel_validation = True
        self.max_workers = config.inference.num_workers if hasattr(config, 'inference') else 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Validation rules registry
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.protocol_rules: Dict[str, List[ValidationRule]] = defaultdict(list)
        
        # Parsers and grammars
        self.parsers: Dict[str, GeneratedParser] = {}
        self.grammars: Dict[str, Grammar] = {}
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_validation_time': 0.0,
            'issue_counts_by_severity': defaultdict(int)
        }
        
        # Initialize built-in validation rules
        self._initialize_builtin_rules()
        
        self.logger.info("Message Validator initialized")

    def _initialize_builtin_rules(self) -> None:
        """Initialize built-in validation rules."""
        # Basic structure validation rules
        self.add_validation_rule(ValidationRule(
            name="message_not_empty",
            description="Message must not be empty",
            validator_func=self._validate_not_empty,
            severity=ValidationSeverity.ERROR
        ))
        
        self.add_validation_rule(ValidationRule(
            name="message_size_limit",
            description="Message size must be within limits",
            validator_func=self._validate_size_limit,
            severity=ValidationSeverity.WARNING
        ))
        
        self.add_validation_rule(ValidationRule(
            name="encoding_consistency",
            description="Message encoding must be consistent",
            validator_func=self._validate_encoding_consistency,
            severity=ValidationSeverity.WARNING
        ))
        
        # Protocol-specific rules
        self.add_validation_rule(ValidationRule(
            name="http_headers_valid",
            description="HTTP headers must be properly formatted",
            validator_func=self._validate_http_headers,
            severity=ValidationSeverity.ERROR,
            protocol_specific="http"
        ))
        
        self.add_validation_rule(ValidationRule(
            name="json_syntax_valid",
            description="JSON syntax must be valid",
            validator_func=self._validate_json_syntax,
            severity=ValidationSeverity.ERROR,
            protocol_specific="json"
        ))
        
        self.add_validation_rule(ValidationRule(
            name="xml_syntax_valid",
            description="XML syntax must be valid",
            validator_func=self._validate_xml_syntax,
            severity=ValidationSeverity.ERROR,
            protocol_specific="xml"
        ))
        
        # Security validation rules
        self.add_validation_rule(ValidationRule(
            name="no_null_bytes",
            description="Message should not contain unexpected null bytes",
            validator_func=self._validate_no_unexpected_nulls,
            severity=ValidationSeverity.WARNING
        ))
        
        self.add_validation_rule(ValidationRule(
            name="control_character_check",
            description="Check for suspicious control characters",
            validator_func=self._validate_control_characters,
            severity=ValidationSeverity.INFO
        ))
        
        # Integrity validation rules
        self.add_validation_rule(ValidationRule(
            name="checksum_validation",
            description="Validate message checksums if present",
            validator_func=self._validate_checksums,
            severity=ValidationSeverity.ERROR
        ))
        
        self.add_validation_rule(ValidationRule(
            name="length_field_consistency",
            description="Length fields must match actual lengths",
            validator_func=self._validate_length_fields,
            severity=ValidationSeverity.ERROR
        ))

    async def validate(
        self,
        message: bytes,
        protocol_type: Optional[str] = None,
        validation_level: Optional[ValidationLevel] = None,
        custom_rules: Optional[List[ValidationRule]] = None
    ) -> ValidationResult:
        """
        Validate a protocol message.
        
        Args:
            message: Message data to validate
            protocol_type: Known protocol type (optional)
            validation_level: Validation strictness level
            custom_rules: Additional custom validation rules
            
        Returns:
            Validation result with issues and metadata
        """
        start_time = time.time()
        validation_level = validation_level or self.default_validation_level
        
        self.logger.debug(f"Validating message of {len(message)} bytes, protocol: {protocol_type}")
        
        try:
            # Update statistics
            self.validation_stats['total_validations'] += 1
            
            # Basic pre-validation checks
            if len(message) > self.max_message_size:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    issues=[ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        code="message_too_large",
                        message=f"Message size {len(message)} exceeds limit {self.max_message_size}",
                        metadata={'actual_size': len(message), 'max_size': self.max_message_size}
                    )]
                )
            
            # Collect all applicable validation rules
            applicable_rules = await self._get_applicable_rules(protocol_type, validation_level, custom_rules)
            
            # Execute validation rules
            all_issues = []
            
            if self.use_parallel_validation and len(applicable_rules) > 5:
                # Parallel validation for many rules
                issues_lists = await asyncio.gather(*[
                    self._execute_rule_async(rule, message, {'protocol_type': protocol_type})
                    for rule in applicable_rules
                ], return_exceptions=True)
                
                for issues in issues_lists:
                    if isinstance(issues, list):
                        all_issues.extend(issues)
                    elif isinstance(issues, Exception):
                        self.logger.error(f"Validation rule failed: {issues}")
                        all_issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code="validation_rule_error",
                            message=f"Validation rule execution failed: {issues}"
                        ))
            else:
                # Sequential validation
                for rule in applicable_rules:
                    try:
                        issues = await self._execute_rule_async(rule, message, {'protocol_type': protocol_type})
                        all_issues.extend(issues)
                    except Exception as e:
                        self.logger.error(f"Validation rule {rule.name} failed: {e}")
                        all_issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code="validation_rule_error",
                            message=f"Rule '{rule.name}' execution failed: {e}"
                        ))
            
            # Grammar-based validation if available
            if protocol_type and protocol_type in self.grammars:
                grammar_issues = await self._validate_against_grammar(message, protocol_type)
                all_issues.extend(grammar_issues)
            
            # Parser-based validation if available
            if protocol_type and protocol_type in self.parsers:
                parser_issues = await self._validate_with_parser(message, protocol_type)
                all_issues.extend(parser_issues)
            
            # Calculate overall validity and confidence
            is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                             for issue in all_issues)
            
            confidence = await self._calculate_validation_confidence(all_issues, validation_level)
            
            # Update statistics
            if is_valid:
                self.validation_stats['successful_validations'] += 1
            else:
                self.validation_stats['failed_validations'] += 1
            
            for issue in all_issues:
                self.validation_stats['issue_counts_by_severity'][issue.severity.value] += 1
            
            validation_time = time.time() - start_time
            self.validation_stats['average_validation_time'] = (
                (self.validation_stats['average_validation_time'] * (self.validation_stats['total_validations'] - 1) + validation_time) /
                self.validation_stats['total_validations']
            )
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                issues=all_issues,
                metadata={
                    'validation_time': validation_time,
                    'validation_level': validation_level.value,
                    'protocol_type': protocol_type,
                    'message_size': len(message),
                    'rules_applied': len(applicable_rules)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    code="validation_error",
                    message=f"Validation process failed: {e}"
                )],
                metadata={'error': str(e)}
            )

    async def _get_applicable_rules(
        self,
        protocol_type: Optional[str],
        validation_level: ValidationLevel,
        custom_rules: Optional[List[ValidationRule]]
    ) -> List[ValidationRule]:
        """Get all applicable validation rules."""
        applicable_rules = []
        
        # Add general rules
        for rule in self.validation_rules.values():
            if rule.enabled and rule.protocol_specific is None:
                applicable_rules.append(rule)
        
        # Add protocol-specific rules
        if protocol_type and protocol_type in self.protocol_rules:
            applicable_rules.extend(self.protocol_rules[protocol_type])
        
        # Add custom rules
        if custom_rules:
            applicable_rules.extend(custom_rules)
        
        # Filter by validation level
        if validation_level == ValidationLevel.BASIC:
            applicable_rules = [r for r in applicable_rules 
                              if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        elif validation_level == ValidationLevel.STANDARD:
            applicable_rules = [r for r in applicable_rules 
                              if r.severity != ValidationSeverity.INFO]
        # STRICT and ENTERPRISE levels include all rules
        
        return applicable_rules

    async def _execute_rule_async(
        self, 
        rule: ValidationRule, 
        message: bytes, 
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Execute a validation rule asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            rule.validator_func, 
            message, 
            context
        )

    async def _calculate_validation_confidence(
        self, 
        issues: List[ValidationIssue], 
        validation_level: ValidationLevel
    ) -> float:
        """Calculate confidence score for validation result."""
        if not issues:
            return 1.0
        
        # Severity weights
        severity_weights = {
            ValidationSeverity.INFO: 0.05,
            ValidationSeverity.WARNING: 0.15,
            ValidationSeverity.ERROR: 0.5,
            ValidationSeverity.CRITICAL: 1.0
        }
        
        # Calculate penalty based on issues
        total_penalty = sum(severity_weights[issue.severity] for issue in issues)
        
        # Adjust based on validation level
        level_multipliers = {
            ValidationLevel.BASIC: 0.8,
            ValidationLevel.STANDARD: 1.0,
            ValidationLevel.STRICT: 1.2,
            ValidationLevel.ENTERPRISE: 1.5
        }
        
        adjusted_penalty = total_penalty * level_multipliers[validation_level]
        confidence = max(0.0, 1.0 - (adjusted_penalty / len(issues) if issues else 1.0))
        
        return confidence

    # Built-in validation rule implementations

    def _validate_not_empty(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate message is not empty."""
        if len(message) == 0:
            return [ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="empty_message",
                message="Message cannot be empty"
            )]
        return []

    def _validate_size_limit(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate message size is reasonable."""
        issues = []
        
        # Check for very small messages (might be incomplete)
        if len(message) < 4:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="message_too_small",
                message="Message appears very small, might be incomplete",
                actual_value=len(message),
                suggestion="Verify message is complete"
            ))
        
        # Check for very large messages
        if len(message) > 1024 * 1024:  # 1MB
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="message_very_large",
                message="Message is unusually large",
                actual_value=len(message),
                suggestion="Consider if message size is appropriate"
            ))
        
        return issues

    def _validate_encoding_consistency(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate encoding consistency."""
        issues = []
        
        # Check for mixed encoding indicators
        has_utf8_bom = message.startswith(b'\xEF\xBB\xBF')
        has_ascii_chars = any(32 <= b <= 126 for b in message)
        has_high_bytes = any(b > 127 for b in message)
        
        if has_ascii_chars and has_high_bytes and not has_utf8_bom:
            # Try to decode as UTF-8
            try:
                message.decode('utf-8')
            except UnicodeDecodeError:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="encoding_inconsistency",
                    message="Message contains mixed ASCII and high bytes but is not valid UTF-8",
                    suggestion="Verify correct encoding"
                ))
        
        return issues

    def _validate_http_headers(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate HTTP headers format."""
        issues = []
        
        try:
            text = message.decode('utf-8', errors='ignore')
            
            # Look for HTTP header patterns
            if 'HTTP/' in text or any(text.startswith(method) for method in ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS']):
                lines = text.split('\n')
                
                # Validate header format
                for i, line in enumerate(lines[1:], 1):  # Skip first line (request/status line)
                    if line.strip() == '':
                        break  # End of headers
                    
                    if ':' not in line:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="invalid_http_header",
                            message=f"Invalid HTTP header format at line {i}",
                            position=i,
                            actual_value=line.strip(),
                            suggestion="Headers must be in 'Name: Value' format"
                        ))
        
        except Exception:
            pass  # Not HTTP or decoding failed
        
        return issues

    def _validate_json_syntax(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate JSON syntax."""
        issues = []
        
        try:
            import json
            text = message.decode('utf-8')
            
            # Check if it looks like JSON
            if text.strip().startswith(('{', '[')):
                try:
                    json.loads(text)
                except json.JSONDecodeError as e:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="invalid_json_syntax",
                        message=f"Invalid JSON syntax: {e.msg}",
                        position=e.pos if hasattr(e, 'pos') else None,
                        suggestion="Fix JSON syntax errors"
                    ))
        
        except UnicodeDecodeError:
            pass  # Not text
        
        return issues

    def _validate_xml_syntax(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate XML syntax."""
        issues = []
        
        try:
            import xml.etree.ElementTree as ET
            text = message.decode('utf-8')
            
            # Check if it looks like XML
            if text.strip().startswith('<'):
                try:
                    ET.fromstring(text)
                except ET.ParseError as e:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="invalid_xml_syntax",
                        message=f"Invalid XML syntax: {e}",
                        suggestion="Fix XML syntax errors"
                    ))
        
        except UnicodeDecodeError:
            pass  # Not text
        
        return issues

    def _validate_no_unexpected_nulls(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate no unexpected null bytes."""
        issues = []
        
        null_count = message.count(b'\x00')
        if null_count > 0:
            # Check if nulls are at expected positions (e.g., string terminators)
            null_positions = [i for i, b in enumerate(message) if b == 0]
            
            # Heuristic: nulls at the end are often legitimate
            unexpected_nulls = [pos for pos in null_positions if pos < len(message) - null_count]
            
            if unexpected_nulls:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="unexpected_null_bytes",
                    message=f"Found {len(unexpected_nulls)} unexpected null bytes",
                    metadata={'positions': unexpected_nulls[:10]},  # Limit to first 10
                    suggestion="Verify null bytes are intentional"
                ))
        
        return issues

    def _validate_control_characters(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate control characters usage."""
        issues = []
        
        # Count various control characters
        control_chars = {}
        for b in message:
            if b < 32 and b not in [9, 10, 13]:  # Exclude tab, LF, CR
                control_chars[b] = control_chars.get(b, 0) + 1
        
        if control_chars:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="control_characters_present",
                message=f"Found control characters: {list(control_chars.keys())}",
                metadata={'control_chars': control_chars},
                suggestion="Verify control characters are expected"
            ))
        
        return issues

    def _validate_checksums(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate checksums if present."""
        issues = []
        
        # This is a simplified implementation
        # In practice, you'd need protocol-specific checksum validation
        
        # Look for common checksum patterns
        if len(message) >= 4:
            # Check for potential CRC at the end
            potential_crc = struct.unpack('>I', message[-4:])[0]
            calculated_crc = zlib.crc32(message[:-4]) & 0xffffffff
            
            if potential_crc == calculated_crc:
                # CRC matches - this is good
                pass
            elif len(message) > 10:  # Only report if message is substantial
                # This might not be a CRC, so just info level
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code="potential_checksum_mismatch",
                    message="Last 4 bytes might be a CRC but don't match calculated value",
                    expected_value=calculated_crc,
                    actual_value=potential_crc,
                    suggestion="Verify if message contains checksum"
                ))
        
        return issues

    def _validate_length_fields(self, message: bytes, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate length field consistency."""
        issues = []
        
        if len(message) < 4:
            return issues
        
        # Check common length field patterns
        # Pattern 1: First 2 bytes as big-endian length
        try:
            length_be = struct.unpack('>H', message[:2])[0]
            if length_be + 2 == len(message):
                # Length field matches - good
                pass
            elif length_be + 2 > len(message):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="length_field_exceeds_message",
                    message="Length field indicates more data than present",
                    expected_value=length_be + 2,
                    actual_value=len(message),
                    position=0
                ))
        except struct.error:
            pass
        
        # Pattern 2: First 4 bytes as big-endian length
        try:
            length_be32 = struct.unpack('>I', message[:4])[0]
            if length_be32 + 4 == len(message):
                # Length field matches - good
                pass
            elif length_be32 + 4 > len(message) and length_be32 < 10000:  # Reasonable size
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="length_field32_exceeds_message",
                    message="32-bit length field indicates more data than present",
                    expected_value=length_be32 + 4,
                    actual_value=len(message),
                    position=0
                ))
        except struct.error:
            pass
        
        return issues

    async def _validate_against_grammar(self, message: bytes, protocol_type: str) -> List[ValidationIssue]:
        """Validate message against learned grammar."""
        issues = []
        
        try:
            grammar = self.grammars[protocol_type]
            
            # This is a simplified grammar validation
            # In practice, you'd use a proper parser to validate against the grammar
            
            # For now, just check if message contains expected patterns
            # This would be replaced with actual grammar-based parsing
            
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="grammar_validation_placeholder",
                message="Grammar validation not yet implemented",
                suggestion="Use parser-based validation instead"
            ))
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="grammar_validation_error",
                message=f"Grammar validation failed: {e}"
            ))
        
        return issues

    async def _validate_with_parser(self, message: bytes, protocol_type: str) -> List[ValidationIssue]:
        """Validate message using generated parser."""
        issues = []
        
        try:
            parser = self.parsers[protocol_type]
            parse_result = await parser.parse_function(message)
            
            if not parse_result.success:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="parser_validation_failed",
                    message="Message failed to parse with generated parser",
                    metadata={
                        'parse_errors': parse_result.errors,
                        'confidence': parse_result.confidence
                    },
                    suggestion="Check message format against protocol specification"
                ))
            elif parse_result.confidence < 0.7:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="low_parse_confidence",
                    message=f"Parser confidence is low: {parse_result.confidence:.2f}",
                    actual_value=parse_result.confidence,
                    suggestion="Message may not fully conform to expected format"
                ))
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="parser_validation_error",
                message=f"Parser validation failed: {e}"
            ))
        
        return issues

    # Public API methods

    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.validation_rules[rule.name] = rule
        
        if rule.protocol_specific:
            self.protocol_rules[rule.protocol_specific].append(rule)
        
        self.logger.debug(f"Added validation rule: {rule.name}")

    def remove_validation_rule(self, rule_name: str) -> bool:
        """Remove a validation rule."""
        if rule_name in self.validation_rules:
            rule = self.validation_rules[rule_name]
            del self.validation_rules[rule_name]
            
            # Remove from protocol-specific rules
            if rule.protocol_specific and rule.protocol_specific in self.protocol_rules:
                self.protocol_rules[rule.protocol_specific] = [
                    r for r in self.protocol_rules[rule.protocol_specific] 
                    if r.name != rule_name
                ]
            
            self.logger.debug(f"Removed validation rule: {rule_name}")
            return True
        
        return False

    def enable_rule(self, rule_name: str) -> bool:
        """Enable a validation rule."""
        if rule_name in self.validation_rules:
            self.validation_rules[rule_name].enabled = True
            return True
        return False

    def disable_rule(self, rule_name: str) -> bool:
        """Disable a validation rule."""
        if rule_name in self.validation_rules:
            self.validation_rules[rule_name].enabled = False
            return True
        return False

    def register_parser(self, protocol_type: str, parser: GeneratedParser) -> None:
        """Register a parser for validation."""
        self.parsers[protocol_type] = parser
        self.logger.info(f"Registered parser for protocol: {protocol_type}")

    def register_grammar(self, protocol_type: str, grammar: Grammar) -> None:
        """Register a grammar for validation."""
        self.grammars[protocol_type] = grammar
        self.logger.info(f"Registered grammar for protocol: {protocol_type}")

    def get_validation_rules(self, protocol_type: Optional[str] = None) -> List[ValidationRule]:
        """Get list of validation rules."""
        if protocol_type:
            return [rule for rule in self.validation_rules.values() 
                   if rule.protocol_specific is None or rule.protocol_specific == protocol_type]
        else:
            return list(self.validation_rules.values())

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return dict(self.validation_stats)

    async def bulk_validate(
        self,
        messages: List[Tuple[bytes, Optional[str]]],  # (message, protocol_type)
        validation_level: Optional[ValidationLevel] = None
    ) -> List[ValidationResult]:
        """Validate multiple messages in parallel."""
        self.logger.info(f"Bulk validating {len(messages)} messages")
        
        validation_tasks = [
            self.validate(message, protocol_type, validation_level)
            for message, protocol_type in messages
        ]
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    issues=[ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        code="bulk_validation_error",
                        message=f"Validation failed for message {i}: {result}"
                    )]
                ))
            else:
                final_results.append(result)
        
        return final_results

    async def validate_stream(
        self,
        message_stream: asyncio.StreamReader,
        protocol_type: Optional[str] = None,
        max_messages: int = 1000
    ) -> AsyncIterator[ValidationResult]:
        """Validate messages from a stream."""
        message_count = 0
        
        while message_count < max_messages:
            try:
                # Read message (this is protocol-specific)
                # For demonstration, reading line by line
                line = await message_stream.readline()
                if not line:
                    break
                
                result = await self.validate(line.rstrip(), protocol_type)
                yield result
                
                message_count += 1
                
            except Exception as e:
                yield ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    issues=[ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        code="stream_validation_error",
                        message=f"Stream validation error: {e}"
                    )]
                )
                break

    async def shutdown(self):
        """Shutdown validator and cleanup resources."""
        self.logger.info("Shutting down Message Validator")
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Clear registries
        self.validation_rules.clear()
        self.protocol_rules.clear()
        self.parsers.clear()
        self.grammars.clear()
        
        self.logger.info("Message Validator shutdown completed")