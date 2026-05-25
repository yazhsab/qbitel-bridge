"""
Client codebase PQC migration reviewer.

This module scans client source repositories for classical cryptographic usage,
turns each finding into a QBITEL CryptoAsset, scores it with the existing
QuantumThreatScorer, and builds a prioritized migration plan.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

from .quantum_threat_scoring import (
    CryptoAsset,
    DataSensitivity,
    MigrationPhase,
    PortfolioAssessment,
    QuantumThreatScorer,
    RiskLevel,
)

logger = logging.getLogger(__name__)


DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "target",
        ".pytest_cache",
        ".mypy_cache",
    }
)

DEFAULT_EXTENSIONS = frozenset(
    {
        ".py",
        ".java",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".cs",
        ".cpp",
        ".cc",
        ".c",
        ".h",
        ".hpp",
        ".kt",
        ".scala",
        ".rb",
        ".php",
        ".cbl",
        ".cob",
        ".cpy",
        ".yaml",
        ".yml",
        ".xml",
        ".properties",
        ".conf",
        ".ini",
        ".tf",
    }
)

EXTENSION_LANGUAGE = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".kt": "kotlin",
    ".scala": "scala",
    ".rb": "ruby",
    ".php": "php",
    ".cbl": "cobol",
    ".cob": "cobol",
    ".cpy": "cobol",
    ".yaml": "config",
    ".yml": "config",
    ".xml": "config",
    ".properties": "config",
    ".conf": "config",
    ".ini": "config",
    ".tf": "terraform",
}


@dataclass(frozen=True)
class CryptoDetectionPattern:
    """Regex-backed detector for one cryptographic usage pattern."""

    name: str
    regex: re.Pattern[str]
    algorithm: str
    category: str
    key_size_bits: int
    confidence: float
    recommended_algorithm: str
    is_key_exchange: bool = False


@dataclass
class CodebaseScanConfig:
    """Configuration for client codebase scanning."""

    include_extensions: Sequence[str] = field(default_factory=lambda: sorted(DEFAULT_EXTENSIONS))
    excluded_dirs: Sequence[str] = field(default_factory=lambda: sorted(DEFAULT_EXCLUDED_DIRS))
    max_file_size_bytes: int = 1_000_000
    data_sensitivity: DataSensitivity = DataSensitivity.CONFIDENTIAL
    data_retention_years: int = 10
    domain: str = "client-codebase"
    current_migration_phase: MigrationPhase = MigrationPhase.NOT_STARTED


@dataclass
class CryptoFinding:
    """A detected classical or weak cryptographic usage in source code."""

    finding_id: str
    file_path: str
    line_number: int
    language: str
    category: str
    algorithm: str
    key_size_bits: int
    matched_text: str
    confidence: float
    recommended_algorithm: str
    is_key_exchange: bool
    rationale: str

    def to_asset(self, config: CodebaseScanConfig) -> CryptoAsset:
        return CryptoAsset(
            asset_id=self.finding_id,
            name=f"{self.algorithm} in {self.file_path}:{self.line_number}",
            algorithm=self.algorithm,
            key_size_bits=self.key_size_bits,
            data_sensitivity=config.data_sensitivity,
            data_retention_years=config.data_retention_years,
            system_count=1,
            migration_phase=config.current_migration_phase,
            domain=config.domain,
            is_key_exchange=self.is_key_exchange,
        )


@dataclass
class MigrationStep:
    """One actionable step in the generated PQC migration plan."""

    step_id: str
    priority: int
    title: str
    description: str
    risk_level: str
    affected_findings: List[str]
    target_algorithm: str
    automation_mode: str
    validation_required: List[str]


@dataclass
class PQCOverlayPlan:
    """Non-invasive PQC layer that can be applied around the codebase."""

    mode: str
    components: List[str]
    rollout_steps: List[str]
    zero_touch_ready: bool
    guardrails: List[str]


@dataclass
class CodebaseMigrationReport:
    """Full output of a client codebase PQC review."""

    report_id: str
    scanned_at: float
    scanned_files: int
    skipped_files: int
    findings: List[CryptoFinding]
    portfolio_assessment: PortfolioAssessment
    migration_plan: List[MigrationStep]
    overlay_plan: PQCOverlayPlan
    ai_review_prompt: str

    def to_dict(self) -> Dict[str, object]:
        """Serialize report content into JSON-friendly primitives."""
        return {
            "report_id": self.report_id,
            "scanned_at": self.scanned_at,
            "scanned_files": self.scanned_files,
            "skipped_files": self.skipped_files,
            "finding_count": len(self.findings),
            "findings": [asdict(finding) for finding in self.findings],
            "portfolio_assessment": _portfolio_to_dict(self.portfolio_assessment),
            "migration_plan": [asdict(step) for step in self.migration_plan],
            "overlay_plan": asdict(self.overlay_plan),
            "ai_review_prompt": self.ai_review_prompt,
        }


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


CRYPTO_PATTERNS: Tuple[CryptoDetectionPattern, ...] = (
    CryptoDetectionPattern(
        "python-rsa-generate",
        _compile(r"rsa\.generate_private_key\s*\(.*key_size\s*=\s*(?P<bits>2048|3072|4096)"),
        "RSA-2048",
        "asymmetric-key",
        2048,
        0.98,
        "ML-DSA-65 or ML-KEM-768 depending on signing vs encryption usage",
    ),
    CryptoDetectionPattern(
        "java-rsa-keypair",
        _compile(r"KeyPairGenerator\.getInstance\s*\(\s*[\"']RSA[\"']\s*\)"),
        "RSA-2048",
        "asymmetric-key",
        2048,
        0.9,
        "ML-DSA-65 for signatures, ML-KEM-768 for key establishment",
    ),
    CryptoDetectionPattern(
        "node-rsa-keypair",
        _compile(r"generateKeyPair(?:Sync)?\s*\(\s*[\"']rsa[\"']"),
        "RSA-2048",
        "asymmetric-key",
        2048,
        0.92,
        "ML-DSA-65 for signatures, ML-KEM-768 for key establishment",
    ),
    CryptoDetectionPattern(
        "go-rsa-generate",
        _compile(r"rsa\.GenerateKey\s*\(.*,\s*(?P<bits>2048|3072|4096)"),
        "RSA-2048",
        "asymmetric-key",
        2048,
        0.98,
        "ML-DSA-65 for signatures, ML-KEM-768 for key establishment",
    ),
    CryptoDetectionPattern(
        "rsa-padding-encryption",
        _compile(r"(publicEncrypt|privateDecrypt|RSA/ECB|PKCS1Padding|OAEPWithSHA)"),
        "RSA-2048",
        "public-key-encryption",
        2048,
        0.82,
        "ML-KEM-768 or hybrid KEM envelope",
        is_key_exchange=True,
    ),
    CryptoDetectionPattern(
        "ecdsa-p256",
        _compile(r"(SHA256withECDSA|elliptic\.P256|prime256v1|SECP256R1)"),
        "ECDSA-P256",
        "signature",
        256,
        0.88,
        "ML-DSA-65",
    ),
    CryptoDetectionPattern(
        "ecdsa-p384",
        _compile(r"(SHA384withECDSA|elliptic\.P384|SECP384R1|secp384r1)"),
        "ECDSA-P384",
        "signature",
        384,
        0.9,
        "ML-DSA-87",
    ),
    CryptoDetectionPattern(
        "ecdh-p256",
        _compile(r"(KeyAgreement\.getInstance\s*\(\s*[\"']ECDH[\"']|createECDH\s*\(\s*[\"']prime256v1[\"']|ecdh\.P256)"),
        "ECDH-P256",
        "key-exchange",
        256,
        0.92,
        "X25519-ML-KEM-768 hybrid key exchange",
        is_key_exchange=True,
    ),
    CryptoDetectionPattern(
        "x25519-classical",
        _compile(r"(X25519PrivateKey|X25519PublicKey|x25519|X25519)"),
        "ECDH-P256",
        "key-exchange",
        256,
        0.82,
        "X25519-ML-KEM-768 hybrid key exchange",
        is_key_exchange=True,
    ),
    CryptoDetectionPattern(
        "dh-2048",
        _compile(r"(DiffieHellman|generate_parameters\s*\(.*key_size\s*=\s*2048|DHParameterSpec|createDiffieHellman)"),
        "DH-2048",
        "key-exchange",
        2048,
        0.86,
        "ML-KEM-768 or hybrid KEM key establishment",
        is_key_exchange=True,
    ),
    CryptoDetectionPattern(
        "aes-128",
        _compile(r"(AES-128|AES_128|aes-128|key_size\s*=\s*128|KeyGenerator\.getInstance\s*\(\s*[\"']AES[\"']\s*\).*128)"),
        "AES-128",
        "symmetric-encryption",
        128,
        0.76,
        "AES-256-GCM with PQC-protected key establishment",
    ),
    CryptoDetectionPattern(
        "des-3des",
        _compile(r"(TripleDES|3DES|DESede|DES/ECB|DES/CBC|algorithms\.DES\b)"),
        "AES-128",
        "legacy-symmetric-encryption",
        112,
        0.9,
        "AES-256-GCM with PQC-protected key establishment",
    ),
    CryptoDetectionPattern(
        "sha1-md5-signature",
        _compile(
            r"(SHA1withRSA|MD5withRSA|hashes\.SHA1|hashes\.MD5|"
            r"MessageDigest\.getInstance\s*\(\s*[\"'](?:SHA-1|MD5)[\"'])"
        ),
        "RSA-2048",
        "weak-signature-or-digest",
        2048,
        0.78,
        "ML-DSA-65 plus SHA-384/SHA3-384 domain separation",
    ),
)


class ClientCodebasePQCMigrator:
    """
    Reviews client source code and generates a PQC migration plan.

    The scanner is intentionally conservative: it reports findings and produces
    migration steps, but it does not rewrite cryptographic code in place because
    protocol compatibility, certificates, HSM support, and compliance validation
    need explicit test coverage.
    """

    def __init__(
        self,
        config: Optional[CodebaseScanConfig] = None,
        scorer: Optional[QuantumThreatScorer] = None,
    ):
        self.config = config or CodebaseScanConfig()
        self.scorer = scorer or QuantumThreatScorer()
        self._include_extensions = {ext.lower() for ext in self.config.include_extensions}
        self._excluded_dirs = set(self.config.excluded_dirs)

    def scan_path(self, root_path: str | Path) -> CodebaseMigrationReport:
        """Scan a client repository path and return a migration report."""
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Client codebase path does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Client codebase path must be a directory: {root}")

        findings: List[CryptoFinding] = []
        scanned_files = 0
        skipped_files = 0

        for source_file in self._iter_source_files(root):
            try:
                if source_file.stat().st_size > self.config.max_file_size_bytes:
                    skipped_files += 1
                    continue
                text = source_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                skipped_files += 1
                logger.debug("Skipping unreadable source file %s: %s", source_file, exc)
                continue

            scanned_files += 1
            rel_path = source_file.relative_to(root).as_posix()
            findings.extend(self._scan_text(rel_path, text))

        return self._build_report(findings, scanned_files, skipped_files)

    def scan_git_repository(
        self,
        repository_url: str,
        *,
        ref: Optional[str] = None,
        depth: int = 1,
        access_token: Optional[str] = None,
    ) -> CodebaseMigrationReport:
        """
        Clone and scan a Git repository from GitHub, GitLab, Bitbucket, Azure
        DevOps, or any Git-compatible platform.

        Args:
            repository_url: HTTPS, SSH, or local Git repository URL.
            ref: Optional branch, tag, or commit-ish to checkout.
            depth: Shallow clone depth. Use 0 for a full clone.
            access_token: Optional HTTPS token injected only into the clone URL.
        """
        if not repository_url.strip():
            raise ValueError("repository_url is required")
        if depth < 0:
            raise ValueError("depth must be 0 or greater")

        git = shutil.which("git")
        if not git:
            raise RuntimeError("git executable is required for repository_url scans")

        with tempfile.TemporaryDirectory(prefix="qbitel-pqc-scan-") as tmp_dir:
            clone_dir = Path(tmp_dir) / "repo"
            clone_url = self._with_https_token(repository_url, access_token)
            clone_cmd = [git, "clone", "--quiet"]
            if depth > 0:
                clone_cmd.extend(["--depth", str(depth)])
            if ref:
                clone_cmd.extend(["--branch", ref])
            clone_cmd.extend([clone_url, str(clone_dir)])

            self._run_git(clone_cmd, repository_url)
            if ref and self._looks_like_commit(ref):
                self._run_git([git, "-C", str(clone_dir), "checkout", "--quiet", ref], repository_url)

            return self.scan_path(clone_dir)

    def scan_sources(self, sources: Mapping[str, str]) -> CodebaseMigrationReport:
        """Scan source files supplied by API clients without filesystem access."""
        findings: List[CryptoFinding] = []
        scanned_files = 0
        skipped_files = 0

        for file_path, text in sources.items():
            suffix = Path(file_path).suffix.lower()
            if suffix and suffix not in self._include_extensions:
                skipped_files += 1
                continue
            if len(text.encode("utf-8", errors="ignore")) > self.config.max_file_size_bytes:
                skipped_files += 1
                continue
            scanned_files += 1
            findings.extend(self._scan_text(file_path, text))

        return self._build_report(findings, scanned_files, skipped_files)

    @staticmethod
    def _with_https_token(repository_url: str, access_token: Optional[str]) -> str:
        if not access_token:
            return repository_url

        parsed = urlparse(repository_url)
        if parsed.scheme not in {"http", "https"}:
            return repository_url
        if parsed.username or parsed.password:
            return repository_url

        netloc = f"x-access-token:{access_token}@{parsed.netloc}"
        return urlunparse(parsed._replace(netloc=netloc))

    @staticmethod
    def _run_git(command: List[str], repository_url: str) -> None:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            sanitized = result.stderr.replace(repository_url, "<repository_url>").strip()
            raise RuntimeError(f"Git repository scan failed: {sanitized}")

    @staticmethod
    def _looks_like_commit(ref: str) -> bool:
        return bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", ref))

    def _iter_source_files(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in self._excluded_dirs for part in path.parts):
                continue
            suffix = path.suffix.lower()
            if suffix in self._include_extensions:
                yield path

    def _scan_text(self, file_path: str, text: str) -> List[CryptoFinding]:
        language = EXTENSION_LANGUAGE.get(Path(file_path).suffix.lower(), "unknown")
        findings: List[CryptoFinding] = []
        seen: set[Tuple[int, str, str]] = set()
        lines = text.splitlines()

        for index, line in enumerate(lines):
            line_number = index + 1
            stripped = line.strip()
            if not stripped or self._is_comment_only(stripped, language):
                continue
            context = "\n".join(lines[max(0, index - 2) : index + 3])

            for pattern in CRYPTO_PATTERNS:
                match = pattern.regex.search(line)
                if not match:
                    continue

                algorithm, key_size = self._resolve_algorithm(pattern, match, context)
                dedupe_key = (line_number, pattern.name, algorithm)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                matched_text = match.group(0).strip()
                findings.append(
                    CryptoFinding(
                        finding_id=self._finding_id(file_path, line_number, pattern.name, matched_text),
                        file_path=file_path,
                        line_number=line_number,
                        language=language,
                        category=pattern.category,
                        algorithm=algorithm,
                        key_size_bits=key_size,
                        matched_text=matched_text[:240],
                        confidence=pattern.confidence,
                        recommended_algorithm=pattern.recommended_algorithm,
                        is_key_exchange=pattern.is_key_exchange,
                        rationale=self._rationale(pattern, algorithm),
                    )
                )

        return findings

    def _resolve_algorithm(
        self,
        pattern: CryptoDetectionPattern,
        match: re.Match[str],
        context: str,
    ) -> Tuple[str, int]:
        key_size = pattern.key_size_bits
        bits = match.groupdict().get("bits")
        if bits:
            key_size = int(bits)

        algorithm = pattern.algorithm
        if algorithm.startswith("RSA-"):
            algorithm = f"RSA-{key_size}"

        if pattern.name == "java-rsa-keypair" or pattern.name == "node-rsa-keypair":
            inferred_bits = self._infer_key_size_from_text(context)
            if inferred_bits:
                key_size = inferred_bits
                algorithm = f"RSA-{key_size}"

        return algorithm, key_size

    @staticmethod
    def _infer_key_size_from_text(text: str) -> Optional[int]:
        match = re.search(r"(?:modulusLength|initialize|keySize|key_size)\s*[:=(]\s*(2048|3072|4096)", text)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _is_comment_only(stripped_line: str, language: str) -> bool:
        if stripped_line.startswith(("#", "//", "/*", "*", "--")):
            return True
        if language == "cobol" and len(stripped_line) > 6 and stripped_line[6:7] in {"*", "/"}:
            return True
        return False

    @staticmethod
    def _finding_id(file_path: str, line_number: int, pattern_name: str, matched_text: str) -> str:
        digest = hashlib.sha256(f"{file_path}:{line_number}:{pattern_name}:{matched_text}".encode()).hexdigest()
        return f"crypto-{digest[:16]}"

    @staticmethod
    def _rationale(pattern: CryptoDetectionPattern, algorithm: str) -> str:
        if algorithm.startswith(("RSA", "ECDSA", "ECDH", "DH")):
            return (
                f"{algorithm} depends on integer factorization or discrete logarithms "
                "and should be migrated to PQC or hybrid mode."
            )
        if algorithm == "AES-128":
            return (
                "AES-128 has reduced effective security under Grover-style search; "
                "prefer AES-256 and PQC-protected key exchange."
            )
        return f"{pattern.name} should be reviewed for PQC readiness."

    def _build_report(
        self,
        findings: List[CryptoFinding],
        scanned_files: int,
        skipped_files: int,
    ) -> CodebaseMigrationReport:
        assets = [finding.to_asset(self.config) for finding in findings]
        portfolio = self.scorer.assess_portfolio(assets)
        migration_plan = self._generate_migration_plan(findings, portfolio)
        overlay_plan = self._generate_overlay_plan(findings, portfolio)
        report_id = "pqc-migration-" + hashlib.sha256(
            f"{time.time()}:{scanned_files}:{len(findings)}".encode()
        ).hexdigest()[:16]

        return CodebaseMigrationReport(
            report_id=report_id,
            scanned_at=time.time(),
            scanned_files=scanned_files,
            skipped_files=skipped_files,
            findings=findings,
            portfolio_assessment=portfolio,
            migration_plan=migration_plan,
            overlay_plan=overlay_plan,
            ai_review_prompt=self._build_ai_review_prompt(findings, portfolio),
        )

    def _generate_migration_plan(
        self,
        findings: List[CryptoFinding],
        portfolio: PortfolioAssessment,
    ) -> List[MigrationStep]:
        assessment_by_id = {assessment.asset_id: assessment for assessment in portfolio.assessments}
        sorted_findings = sorted(
            findings,
            key=lambda finding: assessment_by_id[finding.finding_id].quantum_risk_score,
            reverse=True,
        )

        steps: List[MigrationStep] = []
        for priority, finding in enumerate(sorted_findings[:25], start=1):
            assessment = assessment_by_id[finding.finding_id]
            target = self._target_algorithm(finding, assessment.recommended_algorithm)
            steps.append(
                MigrationStep(
                    step_id=f"step-{priority:03d}",
                    priority=priority,
                    title=f"Migrate {finding.algorithm} at {finding.file_path}:{finding.line_number}",
                    description=(
                        f"Replace or wrap {finding.category} usage with {target}. "
                        f"Current QRS is {assessment.quantum_risk_score} and harvest-now risk is "
                        f"{'present' if assessment.harvest_now_risk else 'not indicated'}."
                    ),
                    risk_level=assessment.risk_level.value,
                    affected_findings=[finding.finding_id],
                    target_algorithm=target,
                    automation_mode=self._automation_mode(assessment.risk_level),
                    validation_required=[
                        "unit tests for cryptographic compatibility",
                        "interoperability test with existing peers or clients",
                        "certificate/key lifecycle review",
                        "rollback plan for protocol negotiation failures",
                    ],
                )
            )

        if findings:
            steps.append(
                MigrationStep(
                    step_id=f"step-{len(steps) + 1:03d}",
                    priority=len(steps) + 1,
                    title="Introduce crypto agility policy layer",
                    description=(
                        "Centralize algorithm selection behind QBITEL policy so clients can run hybrid "
                        "classical plus PQC during migration and move to PQC-primary when dependencies are ready."
                    ),
                    risk_level="program",
                    affected_findings=[finding.finding_id for finding in sorted_findings],
                    target_algorithm="policy-controlled ML-KEM/ML-DSA profile",
                    automation_mode="generate_adapter_and_policy_patch",
                    validation_required=[
                        "policy approval",
                        "configuration drift test",
                        "SBOM and evidence pack update",
                    ],
                )
            )

        return steps

    def _generate_overlay_plan(
        self,
        findings: List[CryptoFinding],
        portfolio: PortfolioAssessment,
    ) -> PQCOverlayPlan:
        key_exchange_findings = [finding for finding in findings if finding.is_key_exchange]
        signing_findings = [finding for finding in findings if "signature" in finding.category]
        symmetric_findings = [finding for finding in findings if "symmetric" in finding.category]
        high_risk = portfolio.critical_count + portfolio.high_count

        components = [
            "PQC-aware ingress/egress gateway with hybrid TLS or KEM envelope",
            "crypto-agility policy file generated from scan findings",
            "CI scanner gate to prevent new RSA/ECC/DH introductions",
            "SBOM/evidence export for client compliance review",
        ]
        if key_exchange_findings:
            components.append("hybrid ML-KEM key-establishment wrapper for vulnerable key exchange paths")
        if signing_findings:
            components.append("ML-DSA signing adapter for high-risk authentication flows")
        if symmetric_findings:
            components.append("AES-256-GCM data-encryption adapter with PQC-protected key wrapping")

        return PQCOverlayPlan(
            mode="zero_touch_overlay_first",
            components=components,
            rollout_steps=[
                "clone repository and run full crypto inventory",
                "rank findings by quantum risk score and harvest-now exposure",
                "deploy PQC gateway or service-mesh policy around external traffic",
                "generate crypto-agility adapter patches for code-level findings",
                "run tests and policy checks before automatic pull-request creation",
                "promote from hybrid mode to PQC-primary when interoperability passes",
            ],
            zero_touch_ready=high_risk == 0,
            guardrails=[
                "no silent production crypto replacement without passing tests",
                "human approval required for critical/high-risk in-place rewrites",
                "overlay mode is preferred for unowned, vendor, or legacy code",
                "keys, certificates, HSM integration, and peer compatibility must be validated",
            ],
        )

    @staticmethod
    def _target_algorithm(finding: CryptoFinding, scorer_target: str) -> str:
        if finding.recommended_algorithm:
            return finding.recommended_algorithm
        return scorer_target

    @staticmethod
    def _automation_mode(risk_level: RiskLevel) -> str:
        if risk_level in {RiskLevel.CRITICAL, RiskLevel.HIGH}:
            return "human_approved_patch_generation"
        return "automated_pull_request_candidate"

    @staticmethod
    def _build_ai_review_prompt(
        findings: List[CryptoFinding],
        portfolio: PortfolioAssessment,
    ) -> str:
        if not findings:
            return (
                "No classical cryptographic findings were detected. Review repository architecture "
                "for hidden vendor crypto and runtime TLS termination."
            )

        top_findings = sorted(
            portfolio.assessments,
            key=lambda assessment: assessment.quantum_risk_score,
            reverse=True,
        )[:10]
        finding_by_id = {finding.finding_id: finding for finding in findings}
        lines = [
            "Review these client-code cryptography findings and produce a safe PQC migration pull-request plan.",
            "Prioritize compatibility, hybrid rollout, key management, tests, and compliance evidence.",
        ]
        for assessment in top_findings:
            finding = finding_by_id[assessment.asset_id]
            lines.append(
                f"- {finding.file_path}:{finding.line_number} uses {finding.algorithm} "
                f"({finding.category}), QRS={assessment.quantum_risk_score}, "
                f"target={finding.recommended_algorithm}"
            )
        return "\n".join(lines)


def _portfolio_to_dict(portfolio: PortfolioAssessment) -> Dict[str, object]:
    data = asdict(portfolio)
    for assessment in data["assessments"]:
        assessment["risk_level"] = assessment["risk_level"].value
    for assessment in data["most_vulnerable"]:
        assessment["risk_level"] = assessment["risk_level"].value
    return data
