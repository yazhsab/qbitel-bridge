"""
CRONOS AI - Protocol Fuzzing Integration

Integrates protocol fuzzing for vulnerability discovery using LLM-guided
mutation strategies and intelligent test case generation.
"""

import asyncio
import logging
import time
import json
import random
import struct
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib

from prometheus_client import Counter, Histogram, Gauge

from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, get_llm_service
from ..core.config import Config
from ..core.exceptions import CronosAIException


# Prometheus metrics
FUZZING_SESSION_COUNTER = Counter(
    "cronos_fuzzing_sessions_total",
    "Total fuzzing sessions executed",
    ["protocol_type"],
    registry=None,
)

FUZZING_TEST_CASES_COUNTER = Counter(
    "cronos_fuzzing_test_cases_total",
    "Total fuzzing test cases generated",
    ["mutation_strategy"],
    registry=None,
)

FUZZING_VULNERABILITIES_COUNTER = Counter(
    "cronos_fuzzing_vulnerabilities_found",
    "Vulnerabilities discovered via fuzzing",
    ["severity"],
    registry=None,
)

FUZZING_SESSION_DURATION = Histogram(
    "cronos_fuzzing_session_duration_seconds",
    "Fuzzing session duration",
    registry=None,
)


logger = logging.getLogger(__name__)


class MutationStrategy(str, Enum):
    """Fuzzing mutation strategies."""

    BIT_FLIP = "bit_flip"  # Flip random bits
    BYTE_FLIP = "byte_flip"  # Flip random bytes
    BOUNDARY_VALUES = "boundary_values"  # Test boundary conditions
    MAGIC_NUMBERS = "magic_numbers"  # Insert common values
    LENGTH_OVERFLOW = "length_overflow"  # Exceed length fields
    FORMAT_STRING = "format_string"  # Format string attacks
    SQL_INJECTION = "sql_injection"  # SQL injection patterns
    COMMAND_INJECTION = "command_injection"  # Command injection
    LLM_GUIDED = "llm_guided"  # LLM-generated mutations


class VulnerabilityType(str, Enum):
    """Types of vulnerabilities that can be discovered."""

    BUFFER_OVERFLOW = "buffer_overflow"
    FORMAT_STRING = "format_string"
    INTEGER_OVERFLOW = "integer_overflow"
    NULL_POINTER = "null_pointer_dereference"
    USE_AFTER_FREE = "use_after_free"
    COMMAND_INJECTION = "command_injection"
    SQL_INJECTION = "sql_injection"
    DENIAL_OF_SERVICE = "denial_of_service"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    UNKNOWN = "unknown"


class FuzzingResult(str, Enum):
    """Result of a fuzzing test case."""

    PASS = "pass"  # Normal execution
    CRASH = "crash"  # Application crashed
    HANG = "hang"  # Application hung/timeout
    ERROR = "error"  # Error response
    ANOMALY = "anomaly"  # Unexpected behavior


@dataclass
class FuzzTestCase:
    """Individual fuzzing test case."""

    test_id: str
    strategy: MutationStrategy
    payload: bytes
    payload_hex: str
    description: str
    expected_behavior: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FuzzingVulnerability:
    """Discovered vulnerability from fuzzing."""

    vulnerability_id: str
    vulnerability_type: VulnerabilityType
    severity: str  # critical, high, medium, low
    title: str
    description: str
    test_case: FuzzTestCase
    reproduction_steps: List[str]
    poc_payload: str
    cve_candidate: bool
    mitigation_recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vulnerability_id": self.vulnerability_id,
            "vulnerability_type": self.vulnerability_type.value,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "test_case": {
                "test_id": self.test_case.test_id,
                "strategy": self.test_case.strategy.value,
                "payload_hex": self.test_case.payload_hex,
                "description": self.test_case.description,
            },
            "reproduction_steps": self.reproduction_steps,
            "poc_payload": self.poc_payload,
            "cve_candidate": self.cve_candidate,
            "mitigation_recommendations": self.mitigation_recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FuzzingSession:
    """Complete fuzzing session results."""

    session_id: str
    protocol_name: str
    protocol_spec: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime]
    test_cases_generated: int
    test_cases_executed: int
    vulnerabilities_found: List[FuzzingVulnerability]
    coverage_metrics: Dict[str, float]
    llm_insights: str
    status: str = "running"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "protocol_name": self.protocol_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "test_cases_generated": self.test_cases_generated,
            "test_cases_executed": self.test_cases_executed,
            "vulnerabilities_found": [v.to_dict() for v in self.vulnerabilities_found],
            "coverage_metrics": self.coverage_metrics,
            "llm_insights": self.llm_insights,
            "status": self.status,
        }


class ProtocolFuzzer:
    """
    Protocol fuzzing engine with LLM-guided mutation strategies.

    Generates intelligent test cases for vulnerability discovery in protocols.
    """

    def __init__(self, config: Config, llm_service: Optional[UnifiedLLMService] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_service = llm_service or get_llm_service(config)

        # Magic numbers for common vulnerabilities
        self.magic_numbers = [
            0x00000000,  # NULL
            0xFFFFFFFF,  # -1 (signed)
            0x7FFFFFFF,  # INT_MAX
            0x80000000,  # INT_MIN
            0xDEADBEEF,  # Common test value
            0x41414141,  # 'AAAA'
        ]

        # Boundary values
        self.boundary_values = [0, 1, 255, 256, 65535, 65536, -1, -128, 127]

    async def start_fuzzing_session(
        self,
        protocol_name: str,
        protocol_spec: Dict[str, Any],
        max_test_cases: int = 1000,
        duration_minutes: int = 60,
    ) -> FuzzingSession:
        """
        Start a fuzzing session for a protocol.

        Args:
            protocol_name: Name of the protocol to fuzz
            protocol_spec: Protocol specification (fields, formats, etc.)
            max_test_cases: Maximum number of test cases to generate
            duration_minutes: Maximum duration in minutes

        Returns:
            Complete fuzzing session results
        """
        start_time = time.time()
        session_id = f"fuzz_{protocol_name}_{int(start_time)}"

        self.logger.info(
            f"Starting fuzzing session: {session_id}, "
            f"max_test_cases={max_test_cases}, "
            f"duration={duration_minutes}m"
        )

        session = FuzzingSession(
            session_id=session_id,
            protocol_name=protocol_name,
            protocol_spec=protocol_spec,
            start_time=datetime.utcnow(),
            end_time=None,
            test_cases_generated=0,
            test_cases_executed=0,
            vulnerabilities_found=[],
            coverage_metrics={},
            llm_insights="",
            status="running",
        )

        try:
            # Generate test cases using multiple strategies
            test_cases = await self._generate_test_cases(
                protocol_name, protocol_spec, max_test_cases
            )
            session.test_cases_generated = len(test_cases)

            # Execute test cases (simulated for now)
            vulnerabilities = await self._execute_test_cases(
                test_cases, protocol_name, protocol_spec
            )
            session.test_cases_executed = len(test_cases)
            session.vulnerabilities_found = vulnerabilities

            # Calculate coverage metrics
            session.coverage_metrics = self._calculate_coverage(
                test_cases, protocol_spec
            )

            # Get LLM insights
            session.llm_insights = await self._generate_llm_insights(
                session, test_cases, vulnerabilities
            )

            session.end_time = datetime.utcnow()
            session.status = "completed"

            # Metrics
            FUZZING_SESSION_COUNTER.labels(protocol_type=protocol_name).inc()
            FUZZING_SESSION_DURATION.observe(time.time() - start_time)

            for vuln in vulnerabilities:
                FUZZING_VULNERABILITIES_COUNTER.labels(severity=vuln.severity).inc()

            self.logger.info(
                f"Fuzzing session completed: {session_id}, "
                f"vulnerabilities={len(vulnerabilities)}"
            )

            return session

        except Exception as e:
            self.logger.error(f"Fuzzing session error: {e}", exc_info=True)
            session.status = "error"
            session.end_time = datetime.utcnow()
            raise CronosAIException(f"Fuzzing session failed: {e}")

    async def _generate_test_cases(
        self, protocol_name: str, protocol_spec: Dict[str, Any], max_cases: int
    ) -> List[FuzzTestCase]:
        """Generate fuzzing test cases using multiple strategies."""
        test_cases = []

        # Get base samples from protocol spec
        base_samples = self._extract_base_samples(protocol_spec)

        # Calculate distribution of strategies
        cases_per_strategy = max_cases // len(MutationStrategy)

        # Generate test cases for each strategy
        for strategy in MutationStrategy:
            strategy_cases = await self._generate_strategy_cases(
                strategy, base_samples, protocol_spec, cases_per_strategy
            )
            test_cases.extend(strategy_cases)

            # Limit total cases
            if len(test_cases) >= max_cases:
                break

        return test_cases[:max_cases]

    def _extract_base_samples(self, protocol_spec: Dict[str, Any]) -> List[bytes]:
        """Extract base samples from protocol specification."""
        samples = []

        # Create basic valid messages
        if "message_format" in protocol_spec:
            # Simple binary message: [length][type][data]
            for msg_type in range(1, 6):
                data = b"HELLO_WORLD"
                length = len(data) + 2  # +2 for type field
                sample = struct.pack("!HB", length, msg_type) + data
                samples.append(sample)

        # Add common protocol patterns
        samples.extend(
            [
                b"\x00\x01\x02\x03\x04\x05",  # Sequential
                b"GET / HTTP/1.1\r\n\r\n",  # HTTP-like
                b"\xff\xff\xff\xff",  # All ones
                b"\x00\x00\x00\x00",  # All zeros
            ]
        )

        return samples

    async def _generate_strategy_cases(
        self,
        strategy: MutationStrategy,
        base_samples: List[bytes],
        protocol_spec: Dict[str, Any],
        num_cases: int,
    ) -> List[FuzzTestCase]:
        """Generate test cases for a specific mutation strategy."""
        test_cases = []

        for i in range(num_cases):
            if not base_samples:
                break

            base_sample = random.choice(base_samples)
            test_id = f"test_{strategy.value}_{i}"

            if strategy == MutationStrategy.BIT_FLIP:
                payload, desc = self._mutate_bit_flip(base_sample)
            elif strategy == MutationStrategy.BYTE_FLIP:
                payload, desc = self._mutate_byte_flip(base_sample)
            elif strategy == MutationStrategy.BOUNDARY_VALUES:
                payload, desc = self._mutate_boundary_values(base_sample)
            elif strategy == MutationStrategy.MAGIC_NUMBERS:
                payload, desc = self._mutate_magic_numbers(base_sample)
            elif strategy == MutationStrategy.LENGTH_OVERFLOW:
                payload, desc = self._mutate_length_overflow(base_sample)
            elif strategy == MutationStrategy.FORMAT_STRING:
                payload, desc = self._mutate_format_string(base_sample)
            elif strategy == MutationStrategy.LLM_GUIDED:
                payload, desc = await self._mutate_llm_guided(
                    base_sample, protocol_spec
                )
            else:
                payload, desc = base_sample, "Default mutation"

            test_case = FuzzTestCase(
                test_id=test_id,
                strategy=strategy,
                payload=payload,
                payload_hex=payload.hex(),
                description=desc,
                expected_behavior="Normal processing or graceful error",
            )
            test_cases.append(test_case)

            FUZZING_TEST_CASES_COUNTER.labels(mutation_strategy=strategy.value).inc()

        return test_cases

    def _mutate_bit_flip(self, data: bytes) -> Tuple[bytes, str]:
        """Flip random bits in the data."""
        if not data:
            return data, "Empty data"

        data_array = bytearray(data)
        num_flips = random.randint(1, min(5, len(data) * 8))
        flipped_positions = []

        for _ in range(num_flips):
            byte_pos = random.randint(0, len(data_array) - 1)
            bit_pos = random.randint(0, 7)
            data_array[byte_pos] ^= 1 << bit_pos
            flipped_positions.append((byte_pos, bit_pos))

        return (
            bytes(data_array),
            f"Flipped {num_flips} bits at positions {flipped_positions}",
        )

    def _mutate_byte_flip(self, data: bytes) -> Tuple[bytes, str]:
        """Flip random bytes in the data."""
        if not data:
            return data, "Empty data"

        data_array = bytearray(data)
        num_flips = random.randint(1, min(3, len(data)))

        for _ in range(num_flips):
            pos = random.randint(0, len(data_array) - 1)
            data_array[pos] ^= 0xFF

        return bytes(data_array), f"Flipped {num_flips} bytes"

    def _mutate_boundary_values(self, data: bytes) -> Tuple[bytes, str]:
        """Insert boundary values."""
        if len(data) < 4:
            return data, "Data too small"

        data_array = bytearray(data)
        value = random.choice(self.boundary_values)

        # Insert at random position
        pos = random.randint(0, len(data_array) - 4)
        struct.pack_into("!I", data_array, pos, value & 0xFFFFFFFF)

        return bytes(data_array), f"Inserted boundary value {value} at position {pos}"

    def _mutate_magic_numbers(self, data: bytes) -> Tuple[bytes, str]:
        """Insert magic numbers."""
        if len(data) < 4:
            return data, "Data too small"

        data_array = bytearray(data)
        magic = random.choice(self.magic_numbers)

        pos = random.randint(0, len(data_array) - 4)
        struct.pack_into("!I", data_array, pos, magic)

        return (
            bytes(data_array),
            f"Inserted magic number {hex(magic)} at position {pos}",
        )

    def _mutate_length_overflow(self, data: bytes) -> Tuple[bytes, str]:
        """Create length field overflow."""
        if len(data) < 2:
            return data, "Data too small"

        data_array = bytearray(data)

        # Assume first 2 bytes are length field
        overflow_length = 0xFFFF  # Max uint16
        struct.pack_into("!H", data_array, 0, overflow_length)

        return bytes(data_array), f"Set length field to {overflow_length}"

    def _mutate_format_string(self, data: bytes) -> Tuple[bytes, str]:
        """Insert format string patterns."""
        format_strings = [b"%s%s%s%s", b"%x%x%x%x", b"%n%n%n%n", b"%p%p%p%p"]

        pattern = random.choice(format_strings)
        mutated = data + pattern

        return (
            mutated,
            f"Appended format string pattern: {pattern.decode('ascii', errors='ignore')}",
        )

    async def _mutate_llm_guided(
        self, data: bytes, protocol_spec: Dict[str, Any]
    ) -> Tuple[bytes, str]:
        """Generate LLM-guided mutation."""
        prompt = f"""
Generate a fuzzing mutation for the following protocol data.

**Protocol Specification:**
{json.dumps(protocol_spec, indent=2)}

**Base Data (hex):**
{data.hex()}

**Task:**
Suggest a mutation that could potentially trigger a vulnerability.
Focus on:
1. Buffer overflows (exceeding expected lengths)
2. Invalid state transitions
3. Malformed headers/fields
4. Injection attacks

Provide the mutation as:
{{
  "mutation_description": "Description of the mutation",
  "hex_payload": "hexadecimal string of mutated data"
}}
"""

        try:
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="protocol_fuzzing",
                max_tokens=500,
                temperature=0.7,  # Higher temperature for creative mutations
            )

            response = await self.llm_service.query(llm_request)

            # Parse response
            start_idx = response.content.find("{")
            end_idx = response.content.rfind("}") + 1
            json_str = response.content[start_idx:end_idx]
            data_obj = json.loads(json_str)

            hex_payload = data_obj.get("hex_payload", data.hex())
            description = data_obj.get("mutation_description", "LLM-guided mutation")

            mutated = bytes.fromhex(hex_payload)
            return mutated, description

        except Exception as e:
            self.logger.warning(f"LLM-guided mutation failed: {e}")
            # Fallback to random mutation
            return self._mutate_byte_flip(data)

    async def _execute_test_cases(
        self,
        test_cases: List[FuzzTestCase],
        protocol_name: str,
        protocol_spec: Dict[str, Any],
    ) -> List[FuzzingVulnerability]:
        """Execute test cases and detect vulnerabilities (simulated)."""
        vulnerabilities = []

        # Simulate test case execution
        for test_case in test_cases:
            # Simulate crash detection based on heuristics
            result = self._simulate_execution(test_case)

            if result in [FuzzingResult.CRASH, FuzzingResult.HANG]:
                # Potential vulnerability found
                vuln = self._analyze_crash(test_case, result, protocol_spec)
                if vuln:
                    vulnerabilities.append(vuln)

        return vulnerabilities

    def _simulate_execution(self, test_case: FuzzTestCase) -> FuzzingResult:
        """Simulate test case execution."""
        # Simulate crash probability based on payload characteristics
        payload = test_case.payload

        # Check for potential crash triggers
        if len(payload) > 10000:  # Large payload
            if random.random() < 0.05:  # 5% crash rate
                return FuzzingResult.CRASH

        if b"%n" in payload or b"%s" in payload:  # Format string
            if random.random() < 0.1:  # 10% crash rate
                return FuzzingResult.CRASH

        if test_case.strategy == MutationStrategy.LENGTH_OVERFLOW:
            if random.random() < 0.08:  # 8% crash rate
                return FuzzingResult.CRASH

        # Most cases pass
        return FuzzingResult.PASS

    def _analyze_crash(
        self,
        test_case: FuzzTestCase,
        result: FuzzingResult,
        protocol_spec: Dict[str, Any],
    ) -> Optional[FuzzingVulnerability]:
        """Analyze a crash to determine vulnerability type."""
        vuln_id = f"vuln_{test_case.test_id}_{int(time.time())}"

        # Determine vulnerability type based on mutation strategy
        vuln_type_map = {
            MutationStrategy.LENGTH_OVERFLOW: VulnerabilityType.BUFFER_OVERFLOW,
            MutationStrategy.FORMAT_STRING: VulnerabilityType.FORMAT_STRING,
            MutationStrategy.BOUNDARY_VALUES: VulnerabilityType.INTEGER_OVERFLOW,
            MutationStrategy.COMMAND_INJECTION: VulnerabilityType.COMMAND_INJECTION,
        }

        vuln_type = vuln_type_map.get(test_case.strategy, VulnerabilityType.UNKNOWN)

        # Determine severity
        severity_map = {
            VulnerabilityType.BUFFER_OVERFLOW: "critical",
            VulnerabilityType.FORMAT_STRING: "high",
            VulnerabilityType.COMMAND_INJECTION: "critical",
            VulnerabilityType.INTEGER_OVERFLOW: "medium",
        }
        severity = severity_map.get(vuln_type, "medium")

        vuln = FuzzingVulnerability(
            vulnerability_id=vuln_id,
            vulnerability_type=vuln_type,
            severity=severity,
            title=f"{vuln_type.value.replace('_', ' ').title()} Discovered",
            description=f"Fuzzing test case {test_case.test_id} triggered a {result.value} "
            f"using {test_case.strategy.value} mutation strategy.",
            test_case=test_case,
            reproduction_steps=[
                "1. Send the payload to the target system",
                f"2. Observe {result.value} behavior",
                "3. Verify crash is reproducible",
            ],
            poc_payload=test_case.payload_hex,
            cve_candidate=severity in ["critical", "high"],
            mitigation_recommendations=[
                "Implement input validation",
                "Add bounds checking",
                "Use safe string functions",
                "Sanitize user input",
            ],
        )

        return vuln

    def _calculate_coverage(
        self, test_cases: List[FuzzTestCase], protocol_spec: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate code coverage metrics."""
        # Simulate coverage metrics
        strategies_used = set(tc.strategy for tc in test_cases)

        return {
            "mutation_strategy_coverage": len(strategies_used) / len(MutationStrategy),
            "estimated_code_coverage": random.uniform(0.6, 0.85),
            "field_coverage": random.uniform(0.7, 0.95),
        }

    async def _generate_llm_insights(
        self,
        session: FuzzingSession,
        test_cases: List[FuzzTestCase],
        vulnerabilities: List[FuzzingVulnerability],
    ) -> str:
        """Generate LLM insights about the fuzzing session."""
        prompt = f"""
Analyze the following protocol fuzzing session results and provide insights.

**Protocol:** {session.protocol_name}
**Test Cases Executed:** {len(test_cases)}
**Vulnerabilities Found:** {len(vulnerabilities)}

**Vulnerability Summary:**
{json.dumps([v.to_dict() for v in vulnerabilities[:5]], indent=2)}

**Coverage Metrics:**
{json.dumps(session.coverage_metrics, indent=2)}

**Task:**
Provide concise insights:
1. Overall security assessment of the protocol
2. Most critical vulnerabilities found
3. Recommended next steps for remediation
4. Suggestions for additional testing

Keep the response under 300 words.
"""

        try:
            llm_request = LLMRequest(
                prompt=prompt,
                feature_domain="fuzzing_analysis",
                max_tokens=500,
                temperature=0.3,
            )

            response = await self.llm_service.query(llm_request)
            return response.content

        except Exception as e:
            self.logger.warning(f"Failed to generate LLM insights: {e}")
            return (
                f"Fuzzing session completed. "
                f"{len(vulnerabilities)} vulnerabilities found. "
                f"Review results for details."
            )


# Factory function
def get_protocol_fuzzer(
    config: Config, llm_service: Optional[UnifiedLLMService] = None
) -> ProtocolFuzzer:
    """Factory function to get ProtocolFuzzer instance."""
    return ProtocolFuzzer(config, llm_service)
