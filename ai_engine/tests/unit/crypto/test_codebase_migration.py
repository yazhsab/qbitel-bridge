"""Unit tests for client codebase PQC migration reviewer."""

import shutil
import subprocess

import pytest

from ai_engine.crypto.codebase_migration import ClientCodebasePQCMigrator
from ai_engine.crypto.quantum_threat_scoring import DataSensitivity, RiskLevel


def test_scan_sources_detects_classical_crypto_and_scores_portfolio():
    migrator = ClientCodebasePQCMigrator()
    report = migrator.scan_sources(
        {
            "payments/signing.py": (
                "from cryptography.hazmat.primitives.asymmetric import rsa\n"
                "key = rsa.generate_private_key(public_exponent=65537, key_size=2048)\n"
            ),
            "gateway/kex.js": "crypto.generateKeyPairSync('rsa', { modulusLength: 3072 })\n",
            "README.md": "RSA-2048 mentioned in documentation should not be scanned by default.\n",
        }
    )

    algorithms = {finding.algorithm for finding in report.findings}

    assert report.scanned_files == 2
    assert report.skipped_files == 1
    assert algorithms == {"RSA-2048", "RSA-3072"}
    assert report.portfolio_assessment.total_assets == 2
    assert report.portfolio_assessment.high_count + report.portfolio_assessment.critical_count >= 1
    assert report.migration_plan[0].automation_mode in {
        "human_approved_patch_generation",
        "automated_pull_request_candidate",
    }


def test_scan_path_excludes_dependency_directories(tmp_path):
    app_dir = tmp_path / "app"
    vendor_dir = app_dir / "node_modules" / "dep"
    source_dir = app_dir / "src"
    vendor_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    (vendor_dir / "bad.js").write_text("crypto.generateKeyPairSync('rsa')\n")
    (source_dir / "main.py").write_text("key = rsa.generate_private_key(key_size=2048)\n")

    report = ClientCodebasePQCMigrator().scan_path(app_dir)

    assert report.scanned_files == 1
    assert len(report.findings) == 1
    assert report.findings[0].file_path == "src/main.py"


def test_scan_git_repository_clones_and_scans_local_repo(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git is required")

    repo = tmp_path / "client-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "qbitel@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "QBITEL Test"], cwd=repo, check=True)
    (repo / "service.py").write_text("key = rsa.generate_private_key(key_size=2048)\n")
    subprocess.run(["git", "add", "service.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "seed"], cwd=repo, check=True)

    report = ClientCodebasePQCMigrator().scan_git_repository(str(repo))

    assert report.scanned_files == 1
    assert report.findings[0].algorithm == "RSA-2048"
    assert report.overlay_plan.mode == "zero_touch_overlay_first"


def test_report_serializes_enums_to_json_friendly_values():
    report = ClientCodebasePQCMigrator().scan_sources(
        {"service.java": 'KeyAgreement.getInstance("ECDH");\n'}
    )

    payload = report.to_dict()
    assessment = payload["portfolio_assessment"]["assessments"][0]

    assert isinstance(assessment["risk_level"], str)
    assert assessment["risk_level"] in {level.value for level in RiskLevel}
    assert payload["migration_plan"][0]["target_algorithm"] == "X25519-ML-KEM-768 hybrid key exchange"
    assert payload["overlay_plan"]["components"]


def test_multiline_key_size_context_is_used():
    report = ClientCodebasePQCMigrator().scan_sources(
        {
            "Signer.java": (
                'KeyPairGenerator generator = KeyPairGenerator.getInstance("RSA");\n'
                "generator.initialize(4096);\n"
            )
        }
    )

    assert report.findings[0].algorithm == "RSA-4096"
    assert report.findings[0].key_size_bits == 4096


def test_scan_config_drives_asset_sensitivity():
    from ai_engine.crypto.codebase_migration import CodebaseScanConfig

    migrator = ClientCodebasePQCMigrator(
        config=CodebaseScanConfig(
            data_sensitivity=DataSensitivity.TOP_SECRET,
            data_retention_years=30,
            domain="banking",
        )
    )
    report = migrator.scan_sources(
        {"core.go": "key, err := rsa.GenerateKey(rand.Reader, 2048)\n"}
    )

    assessment = report.portfolio_assessment.assessments[0]
    assert assessment.risk_level == RiskLevel.CRITICAL
    assert assessment.harvest_now_risk is True
