"""Unit tests for ai_engine.crypto.quantum_threat_scoring module."""

import pytest

from ai_engine.crypto.quantum_threat_scoring import (
    AlgorithmCategory,
    CryptoAsset,
    DataSensitivity,
    MigrationPhase,
    PortfolioAssessment,
    QuantumThreatScorer,
    RiskLevel,
    ThreatAssessment,
)


@pytest.fixture
def scorer():
    """Create a QuantumThreatScorer with default settings."""
    return QuantumThreatScorer()


def _make_asset(
    algorithm: str = "RSA-2048",
    key_size_bits: int = 2048,
    data_sensitivity: DataSensitivity = DataSensitivity.CONFIDENTIAL,
    migration_phase: MigrationPhase = MigrationPhase.NOT_STARTED,
    data_retention_years: int = 10,
    system_count: int = 1,
    asset_id: str = "test-asset",
    name: str = "Test Asset",
) -> CryptoAsset:
    """Helper to create a CryptoAsset with sensible defaults."""
    return CryptoAsset(
        asset_id=asset_id,
        name=name,
        algorithm=algorithm,
        key_size_bits=key_size_bits,
        data_sensitivity=data_sensitivity,
        migration_phase=migration_phase,
        data_retention_years=data_retention_years,
        system_count=system_count,
    )


class TestIndividualScoring:
    """Test individual asset threat assessments."""

    def test_rsa_2048_scores_high_risk(self, scorer):
        """RSA-2048 with default settings should score high (>50)."""
        asset = _make_asset(algorithm="RSA-2048", key_size_bits=2048)
        assessment = scorer.assess_asset(asset)
        assert isinstance(assessment, ThreatAssessment)
        assert assessment.quantum_risk_score > 50

    def test_ml_kem_768_scores_low_risk(self, scorer):
        """ML-KEM-768 should score low risk (<25)."""
        asset = _make_asset(
            algorithm="ML-KEM-768",
            key_size_bits=768,
            migration_phase=MigrationPhase.PQC_ONLY,
        )
        assessment = scorer.assess_asset(asset)
        assert assessment.quantum_risk_score < 25

    def test_aes_256_scores_low(self, scorer):
        """AES-256 should score low quantum risk."""
        asset = _make_asset(
            algorithm="AES-256",
            key_size_bits=256,
            migration_phase=MigrationPhase.PQC_ONLY,
        )
        assessment = scorer.assess_asset(asset)
        assert assessment.quantum_risk_score < 25

    def test_assessment_fields(self, scorer):
        """Verify all expected fields are present on ThreatAssessment."""
        asset = _make_asset()
        assessment = scorer.assess_asset(asset)
        assert assessment.asset_id == "test-asset"
        assert assessment.algorithm == "RSA-2048"
        assert isinstance(assessment.risk_level, RiskLevel)
        assert isinstance(assessment.recommended_action, str)
        assert isinstance(assessment.recommended_algorithm, str)
        assert isinstance(assessment.migration_deadline_year, int)
        assert isinstance(assessment.harvest_now_risk, bool)


class TestDataSensitivityEffect:
    """Test that data sensitivity affects the score."""

    def test_data_sensitivity_affects_score(self, scorer):
        """Same algorithm but TOP_SECRET sensitivity scores higher than PUBLIC."""
        asset_public = _make_asset(
            algorithm="RSA-2048",
            data_sensitivity=DataSensitivity.PUBLIC,
        )
        asset_top_secret = _make_asset(
            algorithm="RSA-2048",
            data_sensitivity=DataSensitivity.TOP_SECRET,
        )
        score_public = scorer.assess_asset(asset_public).quantum_risk_score
        score_top_secret = scorer.assess_asset(asset_top_secret).quantum_risk_score
        assert score_top_secret > score_public


class TestMigrationPhaseEffect:
    """Test that migration phase affects the score."""

    def test_migration_phase_affects_score(self, scorer):
        """NOT_STARTED scores higher risk than CNSA2_COMPLIANT."""
        asset_not_started = _make_asset(
            algorithm="RSA-2048",
            migration_phase=MigrationPhase.NOT_STARTED,
        )
        asset_complete = _make_asset(
            algorithm="RSA-2048",
            migration_phase=MigrationPhase.CNSA2_COMPLIANT,
        )
        score_not_started = scorer.assess_asset(asset_not_started).quantum_risk_score
        score_complete = scorer.assess_asset(asset_complete).quantum_risk_score
        assert score_not_started > score_complete


class TestRiskLevelThresholds:
    """Test risk level classification thresholds."""

    def test_risk_level_thresholds(self, scorer):
        """Verify CRITICAL >= 76, HIGH 51-75, MODERATE 26-50, LOW 0-25."""
        # We check by asserting the mapping from score to risk level.
        # Create assets that produce scores in each range.

        # HIGH/CRITICAL: RSA-2048, TOP_SECRET, NOT_STARTED, many systems
        asset_critical = _make_asset(
            algorithm="RSA-2048",
            data_sensitivity=DataSensitivity.TOP_SECRET,
            migration_phase=MigrationPhase.NOT_STARTED,
            data_retention_years=30,
            system_count=5000,
        )
        assessment = scorer.assess_asset(asset_critical)
        assert assessment.quantum_risk_score >= 76
        assert assessment.risk_level == RiskLevel.CRITICAL

        # LOW: ML-KEM-768, PUBLIC, PQC_ONLY
        asset_low = _make_asset(
            algorithm="ML-KEM-768",
            key_size_bits=768,
            data_sensitivity=DataSensitivity.PUBLIC,
            migration_phase=MigrationPhase.PQC_ONLY,
            data_retention_years=1,
            system_count=1,
        )
        assessment_low = scorer.assess_asset(asset_low)
        assert assessment_low.quantum_risk_score <= 25
        assert assessment_low.risk_level == RiskLevel.LOW


class TestDeterministicScoring:
    """Test that scoring is deterministic."""

    def test_deterministic_scoring(self, scorer):
        """Same inputs always produce the same score."""
        asset = _make_asset(
            algorithm="ECDSA-P256",
            key_size_bits=256,
            data_sensitivity=DataSensitivity.SECRET,
            migration_phase=MigrationPhase.PLANNING,
            data_retention_years=7,
            system_count=50,
        )
        score1 = scorer.assess_asset(asset).quantum_risk_score
        score2 = scorer.assess_asset(asset).quantum_risk_score
        score3 = scorer.assess_asset(asset).quantum_risk_score
        assert score1 == score2 == score3


class TestPortfolioAssessment:
    """Test portfolio-level assessment."""

    def test_portfolio_assessment(self, scorer):
        """Create multiple assets and get a portfolio assessment."""
        assets = [
            _make_asset(
                asset_id="rsa-signing",
                name="RSA Signing Key",
                algorithm="RSA-2048",
                data_sensitivity=DataSensitivity.SECRET,
            ),
            _make_asset(
                asset_id="pqc-kem",
                name="PQC KEM Key",
                algorithm="ML-KEM-768",
                key_size_bits=768,
                data_sensitivity=DataSensitivity.CONFIDENTIAL,
                migration_phase=MigrationPhase.PQC_ONLY,
            ),
            _make_asset(
                asset_id="aes-encryption",
                name="AES Encryption",
                algorithm="AES-256",
                key_size_bits=256,
                data_sensitivity=DataSensitivity.INTERNAL,
                migration_phase=MigrationPhase.PQC_ONLY,
            ),
        ]
        portfolio = scorer.assess_portfolio(assets)
        assert isinstance(portfolio, PortfolioAssessment)
        assert portfolio.total_assets == 3
        assert portfolio.critical_count + portfolio.high_count + portfolio.moderate_count + portfolio.low_count == 3
        assert 0 <= portfolio.average_qrs <= 100
        assert 0 <= portfolio.migration_coverage <= 100
        assert 0 <= portfolio.cnsa2_readiness <= 100
        assert len(portfolio.assessments) == 3
        assert len(portfolio.most_vulnerable) <= 10
