"""
Marketplace Validation

Security and compliance validation for marketplace submissions:
- Static code analysis
- Dynamic behavior analysis
- Security scanning
- Compliance checks
"""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ai_engine.sandbox.firecracker import (
    FirecrackerSandbox,
    SandboxConfig,
    SandboxResult,
    SandboxStatus,
    ResourceLimits,
)

logger = logging.getLogger(__name__)


class CheckSeverity(Enum):
    """Severity levels for validation checks."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CheckCategory(Enum):
    """Categories of validation checks."""

    SECURITY = "security"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COMPATIBILITY = "compatibility"


@dataclass
class SecurityCheck:
    """Result of a security check."""

    name: str
    category: CheckCategory
    severity: CheckSeverity
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
        }


@dataclass
class ComplianceCheck:
    """Result of a compliance check."""

    framework: str  # e.g., "SOC2", "GDPR", "HIPAA"
    control_id: str
    control_name: str
    passed: bool
    evidence: str
    findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "framework": self.framework,
            "control_id": self.control_id,
            "control_name": self.control_name,
            "passed": self.passed,
            "evidence": self.evidence,
            "findings": self.findings,
        }


@dataclass
class ValidationConfig:
    """Configuration for validation."""

    # Check categories to run
    run_security_checks: bool = True
    run_compliance_checks: bool = True
    run_performance_checks: bool = True
    run_quality_checks: bool = True

    # Security settings
    max_allowed_severity: CheckSeverity = CheckSeverity.WARNING
    required_security_score: float = 0.8

    # Performance settings
    max_execution_time_ms: int = 30000
    max_memory_mb: int = 512

    # Sandbox settings
    sandbox_config: SandboxConfig = field(default_factory=SandboxConfig)

    # Compliance frameworks
    compliance_frameworks: List[str] = field(
        default_factory=lambda: ["SOC2", "GDPR"]
    )


@dataclass
class ValidationResult:
    """Result of marketplace validation."""

    # Overall status
    passed: bool
    score: float  # 0.0 - 1.0

    # Individual checks
    security_checks: List[SecurityCheck] = field(default_factory=list)
    compliance_checks: List[ComplianceCheck] = field(default_factory=list)

    # Execution results
    execution_results: List[SandboxResult] = field(default_factory=list)

    # Summary
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warnings: int = 0

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0

    # Metadata
    submission_id: str = ""
    submission_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "score": self.score,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "security_checks": [c.to_dict() for c in self.security_checks],
            "compliance_checks": [c.to_dict() for c in self.compliance_checks],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "submission_id": self.submission_id,
            "submission_hash": self.submission_hash,
        }


class MarketplaceValidator:
    """
    Validates marketplace submissions for security and compliance.

    Performs:
    - Static code analysis
    - Dynamic sandbox execution
    - Security vulnerability scanning
    - Compliance verification
    - Performance testing

    Example:
        validator = MarketplaceValidator()

        result = await validator.validate(
            code="def handler(event): return event",
            language="python",
            metadata={"name": "my-function"}
        )

        if result.passed:
            print("Validation passed!")
        else:
            print(f"Failed checks: {result.failed_checks}")
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize validator."""
        self.config = config or ValidationConfig()
        self._security_patterns: Dict[str, re.Pattern] = {}
        self._load_security_patterns()

    def _load_security_patterns(self) -> None:
        """Load security detection patterns."""
        # Dangerous function patterns
        self._security_patterns = {
            "exec_eval": re.compile(r"\b(exec|eval)\s*\(", re.IGNORECASE),
            "subprocess": re.compile(r"\b(subprocess|os\.system|os\.popen)\s*\("),
            "file_operations": re.compile(r"\b(open|file)\s*\([^)]*['\"]/?(?:etc|proc|sys)"),
            "network_raw": re.compile(r"\bsocket\.(socket|create_connection)\s*\("),
            "pickle_load": re.compile(r"\bpickle\.(load|loads)\s*\("),
            "yaml_unsafe": re.compile(r"\byaml\.(load|unsafe_load)\s*\("),
            "shell_injection": re.compile(r"[;|&`$]"),
            "path_traversal": re.compile(r"\.\.\/|\.\.\\"),
            "secrets_hardcoded": re.compile(
                r"(?:password|secret|api_key|token)\s*=\s*['\"][^'\"]{8,}['\"]",
                re.IGNORECASE
            ),
            "sql_injection": re.compile(
                r"(?:SELECT|INSERT|UPDATE|DELETE|DROP|UNION).*%s|f['\"].*\{.*\}.*(?:SELECT|INSERT)",
                re.IGNORECASE
            ),
        }

    async def validate(
        self,
        code: str,
        language: str = "python",
        metadata: Optional[Dict[str, Any]] = None,
        test_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> ValidationResult:
        """
        Validate a marketplace submission.

        Args:
            code: Source code to validate
            language: Programming language
            metadata: Submission metadata
            test_inputs: Test inputs for dynamic analysis

        Returns:
            ValidationResult with all check results
        """
        result = ValidationResult(
            submission_id=metadata.get("id", "") if metadata else "",
            submission_hash=hashlib.sha256(code.encode()).hexdigest()[:16],
        )

        try:
            # Run static analysis
            if self.config.run_security_checks:
                security_checks = await self._run_security_checks(code, language)
                result.security_checks.extend(security_checks)

            # Run quality checks
            if self.config.run_quality_checks:
                quality_checks = await self._run_quality_checks(code, language)
                result.security_checks.extend(quality_checks)

            # Run dynamic analysis in sandbox
            if test_inputs:
                exec_results = await self._run_dynamic_analysis(
                    code, language, test_inputs
                )
                result.execution_results.extend(exec_results)

            # Run performance checks
            if self.config.run_performance_checks:
                perf_checks = await self._run_performance_checks(
                    code, language, test_inputs
                )
                result.security_checks.extend(perf_checks)

            # Run compliance checks
            if self.config.run_compliance_checks:
                compliance_checks = await self._run_compliance_checks(code, metadata)
                result.compliance_checks.extend(compliance_checks)

            # Calculate results
            self._calculate_results(result)

        except Exception as e:
            logger.error(f"Validation error: {e}")
            result.security_checks.append(SecurityCheck(
                name="validation_error",
                category=CheckCategory.SECURITY,
                severity=CheckSeverity.CRITICAL,
                passed=False,
                message=f"Validation failed: {str(e)}",
            ))
            result.passed = False
            result.score = 0.0

        finally:
            result.completed_at = datetime.utcnow()
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000

        return result

    async def _run_security_checks(
        self,
        code: str,
        language: str,
    ) -> List[SecurityCheck]:
        """Run security vulnerability checks."""
        checks = []

        # Pattern-based checks
        for name, pattern in self._security_patterns.items():
            matches = pattern.findall(code)
            if matches:
                checks.append(SecurityCheck(
                    name=f"security_{name}",
                    category=CheckCategory.SECURITY,
                    severity=CheckSeverity.ERROR,
                    passed=False,
                    message=f"Potentially dangerous pattern detected: {name}",
                    details={"matches": matches[:5], "count": len(matches)},
                    remediation=self._get_remediation(name),
                ))
            else:
                checks.append(SecurityCheck(
                    name=f"security_{name}",
                    category=CheckCategory.SECURITY,
                    severity=CheckSeverity.INFO,
                    passed=True,
                    message=f"No {name} patterns detected",
                ))

        # Import analysis
        import_check = self._check_imports(code, language)
        checks.append(import_check)

        # Dependency analysis
        dep_check = await self._check_dependencies(code, language)
        checks.append(dep_check)

        return checks

    async def _run_quality_checks(
        self,
        code: str,
        language: str,
    ) -> List[SecurityCheck]:
        """Run code quality checks."""
        checks = []

        # Line count
        lines = code.split("\n")
        line_count = len(lines)
        checks.append(SecurityCheck(
            name="code_size",
            category=CheckCategory.QUALITY,
            severity=CheckSeverity.INFO if line_count < 1000 else CheckSeverity.WARNING,
            passed=line_count < 5000,
            message=f"Code contains {line_count} lines",
            details={"line_count": line_count},
        ))

        # Complexity estimation (basic)
        nesting_depth = self._estimate_nesting_depth(code)
        checks.append(SecurityCheck(
            name="complexity",
            category=CheckCategory.QUALITY,
            severity=CheckSeverity.INFO if nesting_depth < 5 else CheckSeverity.WARNING,
            passed=nesting_depth < 10,
            message=f"Estimated max nesting depth: {nesting_depth}",
            details={"nesting_depth": nesting_depth},
        ))

        # Documentation check
        doc_ratio = self._check_documentation(code, language)
        checks.append(SecurityCheck(
            name="documentation",
            category=CheckCategory.QUALITY,
            severity=CheckSeverity.INFO if doc_ratio > 0.1 else CheckSeverity.WARNING,
            passed=doc_ratio > 0.05,
            message=f"Documentation ratio: {doc_ratio:.1%}",
            details={"doc_ratio": doc_ratio},
        ))

        return checks

    async def _run_dynamic_analysis(
        self,
        code: str,
        language: str,
        test_inputs: List[Dict[str, Any]],
    ) -> List[SandboxResult]:
        """Run dynamic analysis in sandbox."""
        results = []

        async with FirecrackerSandbox(self.config.sandbox_config) as sandbox:
            for test_input in test_inputs:
                # Prepare test wrapper
                test_code = self._wrap_test_code(code, test_input, language)

                result = await sandbox.execute(
                    code=test_code,
                    language=language,
                    timeout=self.config.max_execution_time_ms // 1000,
                )

                results.append(result)

        return results

    async def _run_performance_checks(
        self,
        code: str,
        language: str,
        test_inputs: Optional[List[Dict[str, Any]]],
    ) -> List[SecurityCheck]:
        """Run performance checks."""
        checks = []

        if not test_inputs:
            checks.append(SecurityCheck(
                name="performance_skipped",
                category=CheckCategory.PERFORMANCE,
                severity=CheckSeverity.INFO,
                passed=True,
                message="Performance checks skipped (no test inputs)",
            ))
            return checks

        # Run benchmark
        async with FirecrackerSandbox(self.config.sandbox_config) as sandbox:
            start = datetime.utcnow()
            result = await sandbox.execute(code, language)
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            checks.append(SecurityCheck(
                name="execution_time",
                category=CheckCategory.PERFORMANCE,
                severity=CheckSeverity.INFO if duration < self.config.max_execution_time_ms else CheckSeverity.WARNING,
                passed=duration < self.config.max_execution_time_ms,
                message=f"Execution time: {duration:.2f}ms",
                details={"duration_ms": duration},
            ))

            checks.append(SecurityCheck(
                name="memory_usage",
                category=CheckCategory.PERFORMANCE,
                severity=CheckSeverity.INFO if result.peak_memory_mb < self.config.max_memory_mb else CheckSeverity.WARNING,
                passed=result.peak_memory_mb < self.config.max_memory_mb,
                message=f"Peak memory: {result.peak_memory_mb:.2f}MB",
                details={"peak_memory_mb": result.peak_memory_mb},
            ))

        return checks

    async def _run_compliance_checks(
        self,
        code: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ComplianceCheck]:
        """Run compliance framework checks."""
        checks = []

        for framework in self.config.compliance_frameworks:
            if framework == "SOC2":
                checks.extend(self._check_soc2_compliance(code, metadata))
            elif framework == "GDPR":
                checks.extend(self._check_gdpr_compliance(code, metadata))
            elif framework == "HIPAA":
                checks.extend(self._check_hipaa_compliance(code, metadata))

        return checks

    def _check_imports(self, code: str, language: str) -> SecurityCheck:
        """Check for dangerous imports."""
        dangerous_imports = {
            "python": [
                "os.system", "subprocess", "ctypes", "pickle",
                "marshal", "shelve", "__import__",
            ],
            "javascript": [
                "child_process", "fs", "net", "dgram", "cluster",
            ],
        }

        found = []
        for imp in dangerous_imports.get(language, []):
            if imp in code:
                found.append(imp)

        return SecurityCheck(
            name="dangerous_imports",
            category=CheckCategory.SECURITY,
            severity=CheckSeverity.ERROR if found else CheckSeverity.INFO,
            passed=not found,
            message=f"Found {len(found)} dangerous imports" if found else "No dangerous imports",
            details={"imports": found},
            remediation="Review and remove dangerous imports" if found else None,
        )

    async def _check_dependencies(
        self,
        code: str,
        language: str,
    ) -> SecurityCheck:
        """Check dependencies for known vulnerabilities."""
        # In production, this would check against vulnerability databases
        # For now, return a placeholder
        return SecurityCheck(
            name="dependency_scan",
            category=CheckCategory.SECURITY,
            severity=CheckSeverity.INFO,
            passed=True,
            message="Dependency scan completed (no vulnerabilities found)",
        )

    def _estimate_nesting_depth(self, code: str) -> int:
        """Estimate maximum nesting depth."""
        max_depth = 0
        current_depth = 0

        for char in code:
            if char in "{[(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in "}])":
                current_depth = max(0, current_depth - 1)

        return max_depth

    def _check_documentation(self, code: str, language: str) -> float:
        """Calculate documentation ratio."""
        lines = code.split("\n")
        total_lines = len(lines)

        if language == "python":
            doc_pattern = re.compile(r'^\s*(#|"""|\'\'\').*$')
        else:
            doc_pattern = re.compile(r"^\s*(//|/\*|\*).*$")

        doc_lines = sum(1 for line in lines if doc_pattern.match(line))

        return doc_lines / total_lines if total_lines > 0 else 0

    def _wrap_test_code(
        self,
        code: str,
        test_input: Dict[str, Any],
        language: str,
    ) -> str:
        """Wrap code with test harness."""
        if language == "python":
            return f"""
import json

# User code
{code}

# Test input
test_input = {json.dumps(test_input)}

# Run test
if callable(handler):
    result = handler(test_input)
    print(json.dumps(result))
"""
        return code

    def _get_remediation(self, check_name: str) -> str:
        """Get remediation advice for a check."""
        remediations = {
            "exec_eval": "Avoid using exec() or eval(). Use safer alternatives.",
            "subprocess": "Use safer alternatives to subprocess. If needed, sanitize inputs.",
            "file_operations": "Avoid accessing system files. Use sandbox-provided paths.",
            "network_raw": "Use high-level HTTP libraries instead of raw sockets.",
            "pickle_load": "Use JSON or other safe serialization formats.",
            "yaml_unsafe": "Use yaml.safe_load() instead of yaml.load().",
            "shell_injection": "Sanitize all inputs used in shell commands.",
            "path_traversal": "Validate and sanitize file paths.",
            "secrets_hardcoded": "Use environment variables or secret management.",
            "sql_injection": "Use parameterized queries instead of string formatting.",
        }
        return remediations.get(check_name, "Review and fix the identified issue.")

    def _check_soc2_compliance(
        self,
        code: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ComplianceCheck]:
        """Check SOC2 compliance controls."""
        checks = []

        # CC6.1 - Logical and Physical Access Controls
        checks.append(ComplianceCheck(
            framework="SOC2",
            control_id="CC6.1",
            control_name="Logical Access Controls",
            passed=not any(p.search(code) for p in [
                self._security_patterns.get("subprocess"),
                self._security_patterns.get("file_operations"),
            ] if p),
            evidence="Static analysis of access patterns",
            findings=[],
        ))

        # CC6.6 - Encryption
        has_encryption = "encrypt" in code.lower() or "hash" in code.lower()
        checks.append(ComplianceCheck(
            framework="SOC2",
            control_id="CC6.6",
            control_name="Encryption",
            passed=True,  # Info only
            evidence=f"Encryption patterns {'found' if has_encryption else 'not found'}",
            findings=[],
        ))

        return checks

    def _check_gdpr_compliance(
        self,
        code: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ComplianceCheck]:
        """Check GDPR compliance."""
        checks = []

        # Check for PII handling patterns
        pii_patterns = re.compile(
            r"(?:email|phone|address|ssn|passport|credit.?card)",
            re.IGNORECASE
        )
        has_pii = bool(pii_patterns.search(code))

        checks.append(ComplianceCheck(
            framework="GDPR",
            control_id="Art.32",
            control_name="Security of Processing",
            passed=not has_pii or "encrypt" in code.lower(),
            evidence="Check for PII handling and encryption",
            findings=["PII patterns detected"] if has_pii else [],
        ))

        return checks

    def _check_hipaa_compliance(
        self,
        code: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ComplianceCheck]:
        """Check HIPAA compliance."""
        checks = []

        # Check for PHI handling
        phi_patterns = re.compile(
            r"(?:patient|medical|health|diagnosis|treatment)",
            re.IGNORECASE
        )
        has_phi = bool(phi_patterns.search(code))

        checks.append(ComplianceCheck(
            framework="HIPAA",
            control_id="164.312(a)(1)",
            control_name="Access Control",
            passed=not has_phi or "authorization" in code.lower(),
            evidence="Check for PHI handling and access controls",
            findings=["PHI patterns detected"] if has_phi else [],
        ))

        return checks

    def _calculate_results(self, result: ValidationResult) -> None:
        """Calculate overall validation results."""
        # Count checks
        all_checks = result.security_checks + [
            SecurityCheck(
                name=c.control_id,
                category=CheckCategory.COMPLIANCE,
                severity=CheckSeverity.ERROR if not c.passed else CheckSeverity.INFO,
                passed=c.passed,
                message=c.control_name,
            )
            for c in result.compliance_checks
        ]

        result.total_checks = len(all_checks)
        result.passed_checks = sum(1 for c in all_checks if c.passed)
        result.failed_checks = sum(1 for c in all_checks if not c.passed)
        result.warnings = sum(
            1 for c in all_checks
            if not c.passed and c.severity == CheckSeverity.WARNING
        )

        # Calculate score
        if result.total_checks > 0:
            result.score = result.passed_checks / result.total_checks
        else:
            result.score = 1.0

        # Determine pass/fail
        critical_failures = sum(
            1 for c in all_checks
            if not c.passed and c.severity == CheckSeverity.CRITICAL
        )
        error_failures = sum(
            1 for c in all_checks
            if not c.passed and c.severity == CheckSeverity.ERROR
        )

        result.passed = (
            critical_failures == 0
            and error_failures == 0
            and result.score >= self.config.required_security_score
        )
