"""
CRONOS AI - LLM Guardrails and Output Validation
Safety checks, content filtering, and output validation for LLM responses.

Features:
- Content safety filtering
- Output format validation
- Hallucination detection
- PII detection and masking
- Compliance checking
- Rate limiting
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Type, Union

from pydantic import BaseModel, Field, ValidationError


class GuardrailType(Enum):
    """Types of guardrails."""
    INPUT = "input"  # Check input before sending to LLM
    OUTPUT = "output"  # Check output from LLM
    BOTH = "both"  # Check both input and output


class SeverityLevel(Enum):
    """Severity levels for guardrail violations."""
    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Log warning but continue
    ERROR = "error"  # Block and return error
    CRITICAL = "critical"  # Block, log, and alert


class ViolationType(Enum):
    """Types of guardrail violations."""
    UNSAFE_CONTENT = "unsafe_content"
    PII_DETECTED = "pii_detected"
    INVALID_FORMAT = "invalid_format"
    HALLUCINATION = "hallucination"
    COMPLIANCE_VIOLATION = "compliance_violation"
    RATE_LIMIT = "rate_limit"
    INJECTION_ATTEMPT = "injection_attempt"
    SENSITIVE_TOPIC = "sensitive_topic"
    SCHEMA_VIOLATION = "schema_violation"


@dataclass
class GuardrailViolation:
    """Represents a guardrail violation."""
    violation_type: ViolationType
    severity: SeverityLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    content_excerpt: str = ""


@dataclass
class GuardrailResult:
    """Result of guardrail checks."""
    passed: bool
    violations: List[GuardrailViolation]
    sanitized_content: Optional[str] = None
    check_time: float = 0.0
    guardrails_checked: List[str] = field(default_factory=list)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 100000
    tokens_per_hour: int = 1000000


class PIIPattern:
    """Common PII patterns for detection."""
    EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_US = re.compile(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    SSN = re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b')
    CREDIT_CARD = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
    IP_ADDRESS = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    API_KEY = re.compile(r'\b(?:api[_-]?key|token|secret|password|credential)s?\s*[:=]\s*[\'"]?[\w\-_]{16,}[\'"]?\b', re.IGNORECASE)
    AWS_KEY = re.compile(r'\bAKIA[0-9A-Z]{16}\b')
    DATE_OF_BIRTH = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b')


class ContentFilter:
    """
    Filters content for safety and compliance.

    Detects:
    - Unsafe/harmful content
    - Injection attempts
    - Sensitive topics
    """

    def __init__(
        self,
        custom_blocked_patterns: Optional[List[Pattern]] = None,
        custom_sensitive_topics: Optional[List[str]] = None,
    ):
        self.logger = logging.getLogger(__name__)

        # Default blocked patterns (injection attempts, etc.)
        self.blocked_patterns = [
            re.compile(r'ignore previous instructions', re.IGNORECASE),
            re.compile(r'disregard all prior', re.IGNORECASE),
            re.compile(r'system:\s*override', re.IGNORECASE),
            re.compile(r'<\s*script\s*>', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'exec\s*\(\s*[\'"]', re.IGNORECASE),
            re.compile(r'eval\s*\(\s*[\'"]', re.IGNORECASE),
            re.compile(r'\{\{\s*system', re.IGNORECASE),
        ]

        if custom_blocked_patterns:
            self.blocked_patterns.extend(custom_blocked_patterns)

        # Sensitive topics that require extra caution
        self.sensitive_topics = [
            "medical advice",
            "legal advice",
            "financial advice",
            "political opinion",
            "religious belief",
            "personal data",
            "exploit",
            "vulnerability",
            "attack vector",
        ]

        if custom_sensitive_topics:
            self.sensitive_topics.extend(custom_sensitive_topics)

    def check(self, content: str) -> List[GuardrailViolation]:
        """Check content for safety violations."""
        violations = []

        # Check for injection attempts
        for pattern in self.blocked_patterns:
            if pattern.search(content):
                violations.append(GuardrailViolation(
                    violation_type=ViolationType.INJECTION_ATTEMPT,
                    severity=SeverityLevel.ERROR,
                    message=f"Potential injection attempt detected",
                    details={"pattern": pattern.pattern},
                    content_excerpt=content[:100],
                ))

        # Check for sensitive topics
        content_lower = content.lower()
        for topic in self.sensitive_topics:
            if topic in content_lower:
                violations.append(GuardrailViolation(
                    violation_type=ViolationType.SENSITIVE_TOPIC,
                    severity=SeverityLevel.WARNING,
                    message=f"Sensitive topic detected: {topic}",
                    details={"topic": topic},
                    content_excerpt=content[:100],
                ))

        return violations


class PIIDetector:
    """
    Detects and masks Personally Identifiable Information.
    """

    def __init__(
        self,
        patterns: Optional[Dict[str, Pattern]] = None,
        mask_char: str = "*",
        mask_length: int = 8,
    ):
        self.logger = logging.getLogger(__name__)
        self.mask_char = mask_char
        self.mask_length = mask_length

        self.patterns = patterns or {
            "email": PIIPattern.EMAIL,
            "phone": PIIPattern.PHONE_US,
            "ssn": PIIPattern.SSN,
            "credit_card": PIIPattern.CREDIT_CARD,
            "ip_address": PIIPattern.IP_ADDRESS,
            "api_key": PIIPattern.API_KEY,
            "aws_key": PIIPattern.AWS_KEY,
            "date_of_birth": PIIPattern.DATE_OF_BIRTH,
        }

    def detect(self, content: str) -> List[GuardrailViolation]:
        """Detect PII in content."""
        violations = []

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(content)
            if matches:
                violations.append(GuardrailViolation(
                    violation_type=ViolationType.PII_DETECTED,
                    severity=SeverityLevel.WARNING,
                    message=f"PII detected: {pii_type} ({len(matches)} instance(s))",
                    details={
                        "pii_type": pii_type,
                        "count": len(matches),
                    },
                ))

        return violations

    def mask(self, content: str) -> str:
        """Mask all detected PII in content."""
        masked = content

        for pii_type, pattern in self.patterns.items():
            def replacer(match):
                original = match.group(0)
                # Keep first and last chars for context
                if len(original) > 4:
                    return original[0] + self.mask_char * self.mask_length + original[-1]
                return self.mask_char * len(original)

            masked = pattern.sub(replacer, masked)

        return masked


class OutputValidator:
    """
    Validates LLM output against schemas and expected formats.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_json(
        self,
        content: str,
        schema: Optional[Dict[str, Any]] = None,
        model: Optional[Type[BaseModel]] = None,
    ) -> Tuple[bool, Optional[Any], List[GuardrailViolation]]:
        """
        Validate that content is valid JSON and optionally matches a schema.

        Returns:
            Tuple of (is_valid, parsed_data, violations)
        """
        violations = []

        # Try to parse JSON
        try:
            # Handle markdown code blocks
            json_content = content
            if "```json" in content:
                match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    json_content = match.group(1)
            elif "```" in content:
                match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    json_content = match.group(1)

            data = json.loads(json_content)

        except json.JSONDecodeError as e:
            violations.append(GuardrailViolation(
                violation_type=ViolationType.INVALID_FORMAT,
                severity=SeverityLevel.ERROR,
                message=f"Invalid JSON: {str(e)}",
                details={"error": str(e)},
                content_excerpt=content[:200],
            ))
            return False, None, violations

        # Validate against Pydantic model
        if model:
            try:
                validated = model.model_validate(data)
                return True, validated, violations
            except ValidationError as e:
                violations.append(GuardrailViolation(
                    violation_type=ViolationType.SCHEMA_VIOLATION,
                    severity=SeverityLevel.ERROR,
                    message=f"Schema validation failed: {str(e)}",
                    details={"errors": e.errors()},
                ))
                return False, data, violations

        # Validate against JSON schema (basic validation)
        if schema:
            schema_errors = self._validate_json_schema(data, schema)
            if schema_errors:
                for error in schema_errors:
                    violations.append(GuardrailViolation(
                        violation_type=ViolationType.SCHEMA_VIOLATION,
                        severity=SeverityLevel.ERROR,
                        message=error,
                    ))
                return False, data, violations

        return True, data, violations

    def _validate_json_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str = "",
    ) -> List[str]:
        """Basic JSON schema validation."""
        errors = []

        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(data, dict):
                errors.append(f"{path or 'root'}: expected object, got {type(data).__name__}")
                return errors

            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    errors.append(f"{path}.{field}: required field missing")

            # Validate properties
            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                if prop in data:
                    errors.extend(self._validate_json_schema(
                        data[prop],
                        prop_schema,
                        f"{path}.{prop}" if path else prop
                    ))

        elif schema_type == "array":
            if not isinstance(data, list):
                errors.append(f"{path or 'root'}: expected array, got {type(data).__name__}")
                return errors

            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    errors.extend(self._validate_json_schema(
                        item,
                        items_schema,
                        f"{path}[{i}]"
                    ))

        elif schema_type == "string":
            if not isinstance(data, str):
                errors.append(f"{path or 'root'}: expected string, got {type(data).__name__}")

        elif schema_type == "number" or schema_type == "integer":
            if not isinstance(data, (int, float)):
                errors.append(f"{path or 'root'}: expected number, got {type(data).__name__}")

        elif schema_type == "boolean":
            if not isinstance(data, bool):
                errors.append(f"{path or 'root'}: expected boolean, got {type(data).__name__}")

        return errors

    def validate_format(
        self,
        content: str,
        expected_format: str,
    ) -> List[GuardrailViolation]:
        """Validate content matches expected format."""
        violations = []

        if expected_format == "json":
            is_valid, _, json_violations = self.validate_json(content)
            violations.extend(json_violations)

        elif expected_format == "markdown":
            # Basic markdown validation
            if not any(marker in content for marker in ["#", "-", "*", "```", "**"]):
                violations.append(GuardrailViolation(
                    violation_type=ViolationType.INVALID_FORMAT,
                    severity=SeverityLevel.WARNING,
                    message="Content does not appear to be markdown formatted",
                ))

        elif expected_format == "code":
            # Check for code-like patterns
            code_patterns = [
                r'\bdef\s+\w+',  # Python
                r'\bfunction\s+\w+',  # JavaScript
                r'\bclass\s+\w+',  # Classes
                r'[\{\}\(\);]',  # Brackets/semicolons
            ]
            if not any(re.search(p, content) for p in code_patterns):
                violations.append(GuardrailViolation(
                    violation_type=ViolationType.INVALID_FORMAT,
                    severity=SeverityLevel.WARNING,
                    message="Content does not appear to be code",
                ))

        return violations


class HallucinationDetector:
    """
    Detects potential hallucinations in LLM output.

    Uses heuristics and optional fact-checking.
    """

    def __init__(
        self,
        fact_check_func: Optional[Callable] = None,
        confidence_threshold: float = 0.7,
    ):
        self.fact_check = fact_check_func
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

        # Patterns that often indicate uncertainty or fabrication
        self.uncertainty_patterns = [
            re.compile(r'\bi think\b', re.IGNORECASE),
            re.compile(r'\bi believe\b', re.IGNORECASE),
            re.compile(r'\bprobably\b', re.IGNORECASE),
            re.compile(r'\bmight be\b', re.IGNORECASE),
            re.compile(r'\bpossibly\b', re.IGNORECASE),
            re.compile(r'\bI\'m not sure\b', re.IGNORECASE),
            re.compile(r'\bas far as I know\b', re.IGNORECASE),
        ]

        # Patterns that often precede fabricated facts
        self.fabrication_patterns = [
            re.compile(r'according to\s+(?!the\s+(?:documentation|spec|standard|RFC))', re.IGNORECASE),
            re.compile(r'studies show that', re.IGNORECASE),
            re.compile(r'research indicates', re.IGNORECASE),
            re.compile(r'it is well known that', re.IGNORECASE),
        ]

    def detect(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[GuardrailViolation]:
        """Detect potential hallucinations in content."""
        violations = []
        context = context or {}

        # Check for uncertainty patterns
        uncertainty_count = sum(
            1 for p in self.uncertainty_patterns if p.search(content)
        )

        if uncertainty_count > 2:
            violations.append(GuardrailViolation(
                violation_type=ViolationType.HALLUCINATION,
                severity=SeverityLevel.WARNING,
                message=f"High uncertainty detected ({uncertainty_count} indicators)",
                details={"uncertainty_count": uncertainty_count},
            ))

        # Check for fabrication patterns
        for pattern in self.fabrication_patterns:
            if pattern.search(content):
                violations.append(GuardrailViolation(
                    violation_type=ViolationType.HALLUCINATION,
                    severity=SeverityLevel.WARNING,
                    message="Potential unverifiable claim detected",
                    details={"pattern": pattern.pattern},
                ))

        # Check for contradictions with provided context
        if "expected_facts" in context:
            for fact in context["expected_facts"]:
                if fact.lower() not in content.lower():
                    violations.append(GuardrailViolation(
                        violation_type=ViolationType.HALLUCINATION,
                        severity=SeverityLevel.WARNING,
                        message=f"Expected fact not found: {fact[:50]}...",
                        details={"missing_fact": fact},
                    ))

        return violations


class RateLimiter:
    """
    Rate limiting for LLM requests.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.logger = logging.getLogger(__name__)

        # Request tracking
        self._request_times: List[datetime] = []
        self._token_usage: List[Tuple[datetime, int]] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def check_limit(
        self,
        estimated_tokens: int = 0,
    ) -> Tuple[bool, Optional[GuardrailViolation]]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple of (is_allowed, violation if blocked)
        """
        async with self._lock:
            now = datetime.now()

            # Clean old entries
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)

            self._request_times = [t for t in self._request_times if t > minute_ago]
            self._token_usage = [(t, tokens) for t, tokens in self._token_usage if t > hour_ago]

            # Check request limits
            requests_last_minute = len([t for t in self._request_times if t > minute_ago])
            requests_last_hour = len(self._request_times)

            if requests_last_minute >= self.config.requests_per_minute:
                return False, GuardrailViolation(
                    violation_type=ViolationType.RATE_LIMIT,
                    severity=SeverityLevel.ERROR,
                    message=f"Rate limit exceeded: {requests_last_minute}/{self.config.requests_per_minute} requests per minute",
                    details={
                        "limit": self.config.requests_per_minute,
                        "current": requests_last_minute,
                        "window": "minute",
                    },
                )

            if requests_last_hour >= self.config.requests_per_hour:
                return False, GuardrailViolation(
                    violation_type=ViolationType.RATE_LIMIT,
                    severity=SeverityLevel.ERROR,
                    message=f"Rate limit exceeded: {requests_last_hour}/{self.config.requests_per_hour} requests per hour",
                    details={
                        "limit": self.config.requests_per_hour,
                        "current": requests_last_hour,
                        "window": "hour",
                    },
                )

            # Check token limits
            tokens_last_minute = sum(
                tokens for t, tokens in self._token_usage
                if t > minute_ago
            )
            tokens_last_hour = sum(tokens for _, tokens in self._token_usage)

            if tokens_last_minute + estimated_tokens > self.config.tokens_per_minute:
                return False, GuardrailViolation(
                    violation_type=ViolationType.RATE_LIMIT,
                    severity=SeverityLevel.ERROR,
                    message=f"Token limit exceeded: {tokens_last_minute}/{self.config.tokens_per_minute} tokens per minute",
                    details={
                        "limit": self.config.tokens_per_minute,
                        "current": tokens_last_minute,
                        "estimated_new": estimated_tokens,
                    },
                )

            return True, None

    async def record_usage(self, tokens_used: int) -> None:
        """Record a request and its token usage."""
        async with self._lock:
            now = datetime.now()
            self._request_times.append(now)
            self._token_usage.append((now, tokens_used))


class GuardrailManager:
    """
    Central manager for all guardrails.

    Coordinates multiple guardrail checks and provides a unified interface.
    """

    def __init__(
        self,
        enable_content_filter: bool = True,
        enable_pii_detection: bool = True,
        enable_output_validation: bool = True,
        enable_hallucination_detection: bool = True,
        enable_rate_limiting: bool = True,
        rate_limit_config: Optional[RateLimitConfig] = None,
        auto_mask_pii: bool = True,
        block_on_pii: bool = False,
    ):
        self.logger = logging.getLogger(__name__)
        self.auto_mask_pii = auto_mask_pii
        self.block_on_pii = block_on_pii

        # Initialize guardrails
        self.content_filter = ContentFilter() if enable_content_filter else None
        self.pii_detector = PIIDetector() if enable_pii_detection else None
        self.output_validator = OutputValidator() if enable_output_validation else None
        self.hallucination_detector = HallucinationDetector() if enable_hallucination_detection else None
        self.rate_limiter = RateLimiter(rate_limit_config) if enable_rate_limiting else None

    async def check_input(
        self,
        content: str,
        estimated_tokens: int = 0,
    ) -> GuardrailResult:
        """
        Check input before sending to LLM.

        Args:
            content: Input content to check
            estimated_tokens: Estimated tokens for rate limiting

        Returns:
            GuardrailResult with violations and sanitized content
        """
        start_time = time.time()
        violations = []
        guardrails_checked = []
        sanitized = content

        # Rate limit check
        if self.rate_limiter:
            guardrails_checked.append("rate_limiter")
            allowed, violation = await self.rate_limiter.check_limit(estimated_tokens)
            if not allowed and violation:
                violations.append(violation)
                return GuardrailResult(
                    passed=False,
                    violations=violations,
                    sanitized_content=None,
                    check_time=time.time() - start_time,
                    guardrails_checked=guardrails_checked,
                )

        # Content filter
        if self.content_filter:
            guardrails_checked.append("content_filter")
            filter_violations = self.content_filter.check(content)
            violations.extend(filter_violations)

        # PII detection
        if self.pii_detector:
            guardrails_checked.append("pii_detector")
            pii_violations = self.pii_detector.detect(content)
            violations.extend(pii_violations)

            if self.auto_mask_pii and pii_violations:
                sanitized = self.pii_detector.mask(sanitized)

        # Determine if passed
        has_errors = any(v.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL) for v in violations)
        has_blocking_pii = self.block_on_pii and any(v.violation_type == ViolationType.PII_DETECTED for v in violations)

        return GuardrailResult(
            passed=not (has_errors or has_blocking_pii),
            violations=violations,
            sanitized_content=sanitized,
            check_time=time.time() - start_time,
            guardrails_checked=guardrails_checked,
        )

    async def check_output(
        self,
        content: str,
        expected_format: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        response_model: Optional[Type[BaseModel]] = None,
        context: Optional[Dict[str, Any]] = None,
        tokens_used: int = 0,
    ) -> GuardrailResult:
        """
        Check output from LLM.

        Args:
            content: Output content to check
            expected_format: Expected format (json, markdown, code)
            json_schema: JSON schema for validation
            response_model: Pydantic model for validation
            context: Context for hallucination detection
            tokens_used: Tokens used (for rate limiting)

        Returns:
            GuardrailResult with violations
        """
        start_time = time.time()
        violations = []
        guardrails_checked = []
        sanitized = content

        # Record usage for rate limiting
        if self.rate_limiter and tokens_used > 0:
            await self.rate_limiter.record_usage(tokens_used)

        # Content filter
        if self.content_filter:
            guardrails_checked.append("content_filter")
            filter_violations = self.content_filter.check(content)
            violations.extend(filter_violations)

        # PII detection
        if self.pii_detector:
            guardrails_checked.append("pii_detector")
            pii_violations = self.pii_detector.detect(content)
            violations.extend(pii_violations)

            if self.auto_mask_pii and pii_violations:
                sanitized = self.pii_detector.mask(sanitized)

        # Output validation
        if self.output_validator:
            guardrails_checked.append("output_validator")

            if expected_format:
                format_violations = self.output_validator.validate_format(content, expected_format)
                violations.extend(format_violations)

            if expected_format == "json" or json_schema or response_model:
                _, _, json_violations = self.output_validator.validate_json(
                    content,
                    schema=json_schema,
                    model=response_model,
                )
                violations.extend(json_violations)

        # Hallucination detection
        if self.hallucination_detector:
            guardrails_checked.append("hallucination_detector")
            hallucination_violations = self.hallucination_detector.detect(content, context)
            violations.extend(hallucination_violations)

        # Determine if passed
        has_errors = any(v.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL) for v in violations)

        return GuardrailResult(
            passed=not has_errors,
            violations=violations,
            sanitized_content=sanitized,
            check_time=time.time() - start_time,
            guardrails_checked=guardrails_checked,
        )


class GuardedLLMService:
    """
    Wrapper that adds guardrails to any LLM service.

    Usage:
        guarded_service = GuardedLLMService(llm_service, guardrail_config)
        response = await guarded_service.process_request(request)
    """

    def __init__(
        self,
        llm_service,
        guardrail_manager: Optional[GuardrailManager] = None,
        enable_guardrails: bool = True,
        block_on_input_violation: bool = True,
        block_on_output_violation: bool = False,
    ):
        self.llm_service = llm_service
        self.guardrails = guardrail_manager or GuardrailManager()
        self.enable_guardrails = enable_guardrails
        self.block_on_input_violation = block_on_input_violation
        self.block_on_output_violation = block_on_output_violation
        self.logger = logging.getLogger(__name__)

    async def process_request(self, request) -> Any:
        """
        Process LLM request with guardrail checks.

        Args:
            request: LLM request (LLMRequest or similar)

        Returns:
            LLM response (may be modified by guardrails)

        Raises:
            GuardrailException: If guardrails block the request
        """
        if not self.enable_guardrails:
            return await self.llm_service.process_request(request)

        prompt = getattr(request, "prompt", str(request))

        # Check input
        input_result = await self.guardrails.check_input(
            prompt,
            estimated_tokens=len(prompt.split()) * 2,  # Rough estimate
        )

        if not input_result.passed and self.block_on_input_violation:
            error_messages = [v.message for v in input_result.violations if v.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL)]
            raise GuardrailException(
                f"Input blocked by guardrails: {'; '.join(error_messages)}",
                violations=input_result.violations,
            )

        # Use sanitized prompt if available
        if input_result.sanitized_content and hasattr(request, "prompt"):
            request.prompt = input_result.sanitized_content

        # Call LLM
        response = await self.llm_service.process_request(request)

        # Check output
        content = getattr(response, "content", str(response))
        tokens_used = getattr(response, "tokens_used", 0)

        output_result = await self.guardrails.check_output(
            content,
            tokens_used=tokens_used,
        )

        if not output_result.passed:
            self.logger.warning(
                f"Output guardrail violations: {[v.message for v in output_result.violations]}"
            )

            if self.block_on_output_violation:
                error_messages = [v.message for v in output_result.violations if v.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL)]
                raise GuardrailException(
                    f"Output blocked by guardrails: {'; '.join(error_messages)}",
                    violations=output_result.violations,
                )

        # Apply sanitization to response
        if output_result.sanitized_content and hasattr(response, "content"):
            response.content = output_result.sanitized_content

        return response


class GuardrailException(Exception):
    """Exception raised when guardrails block a request."""

    def __init__(self, message: str, violations: List[GuardrailViolation] = None):
        super().__init__(message)
        self.violations = violations or []


# Global guardrail manager instance
_global_guardrail_manager: Optional[GuardrailManager] = None


def get_guardrail_manager() -> GuardrailManager:
    """Get or create global guardrail manager instance."""
    global _global_guardrail_manager
    if _global_guardrail_manager is None:
        _global_guardrail_manager = GuardrailManager()
    return _global_guardrail_manager


def configure_guardrails(**config) -> GuardrailManager:
    """Configure and return global guardrail manager."""
    global _global_guardrail_manager
    _global_guardrail_manager = GuardrailManager(**config)
    return _global_guardrail_manager
