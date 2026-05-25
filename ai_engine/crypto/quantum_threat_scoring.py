"""
AI-Driven Quantum Threat Scoring Engine

Real-time risk scoring system that evaluates the quantum threat level
for cryptographic assets based on:
    1. Algorithm exposure: Which classical algorithms are in use
    2. Key lifetime: How long keys must remain secure (harvest-now-decrypt-later)
    3. Quantum computing progress: Current state of quantum hardware
    4. Data sensitivity: Classification level of protected data
    5. Migration readiness: How far along the PQC migration is

Produces a composite Quantum Risk Score (QRS) from 0-100 that drives
automated migration priority decisions across all domains.

Scoring Model:
    QRS = Σ(w_i × factor_i) where factors include:
    - Algorithm Vulnerability Score (AVS): 0-100 based on algorithm type
    - Time Horizon Risk (THR): Years until quantum threat materializes
    - Data Sensitivity Multiplier (DSM): Based on classification
    - Migration Gap Score (MGS): How far from PQC-ready
    - Exposure Surface (ES): Number of systems using vulnerable algorithms

The model is calibrated against NIST and CNSA 2.0 timelines and can
incorporate real-time signals (quantum computing milestones, CVEs).

References:
    - NIST SP 800-227: Recommendations for Transition to PQC
    - NSA CNSA 2.0: Algorithm suite transition timeline
    - Mosca's Theorem: x + y > z analysis for crypto migration
    - ETSI QSC: Quantum-Safe Cryptography
"""

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

QTS_ASSESSMENTS = Counter("quantum_threat_assessments_total", "Threat assessments performed")
QTS_SCORE = Histogram("quantum_threat_score", "Distribution of threat scores", buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
QTS_CRITICAL_ASSETS = Gauge("quantum_threat_critical_assets", "Number of critical-risk assets")


class AlgorithmCategory(Enum):
    """Cryptographic algorithm categories for vulnerability scoring."""
    RSA = "rsa"
    ECDSA = "ecdsa"
    ECDH = "ecdh"
    DH = "dh"
    DSA = "dsa"
    AES = "aes"
    SHA2 = "sha2"
    SHA3 = "sha3"
    ML_KEM = "ml-kem"
    ML_DSA = "ml-dsa"
    FALCON = "falcon"
    SLH_DSA = "slh-dsa"
    LMS = "lms"
    XMSS = "xmss"
    HYBRID = "hybrid"


class DataSensitivity(Enum):
    """Data sensitivity classification."""
    PUBLIC = 1
    INTERNAL = 2
    CONFIDENTIAL = 3
    SECRET = 4
    TOP_SECRET = 5


class MigrationPhase(Enum):
    """Current PQC migration phase."""
    NOT_STARTED = auto()
    PLANNING = auto()
    TESTING = auto()
    HYBRID_DEPLOYMENT = auto()
    PQC_PRIMARY = auto()
    PQC_ONLY = auto()
    CNSA2_COMPLIANT = auto()


class RiskLevel(Enum):
    """Quantum risk levels."""
    LOW = "low"              # QRS 0-25
    MODERATE = "moderate"    # QRS 26-50
    HIGH = "high"            # QRS 51-75
    CRITICAL = "critical"    # QRS 76-100


# ──────────────────────────────────────────────────────────────────────
# Algorithm vulnerability database
# ──────────────────────────────────────────────────────────────────────

ALGORITHM_VULNERABILITY: Dict[str, Dict[str, Any]] = {
    # Classical — vulnerable to Shor's algorithm
    "RSA-2048": {"category": AlgorithmCategory.RSA, "avs": 85, "shor_vulnerable": True, "grover_impact": "none", "estimated_qubits": 4000},
    "RSA-3072": {"category": AlgorithmCategory.RSA, "avs": 80, "shor_vulnerable": True, "grover_impact": "none", "estimated_qubits": 6000},
    "RSA-4096": {"category": AlgorithmCategory.RSA, "avs": 75, "shor_vulnerable": True, "grover_impact": "none", "estimated_qubits": 8000},
    "ECDSA-P256": {"category": AlgorithmCategory.ECDSA, "avs": 90, "shor_vulnerable": True, "grover_impact": "none", "estimated_qubits": 2500},
    "ECDSA-P384": {"category": AlgorithmCategory.ECDSA, "avs": 85, "shor_vulnerable": True, "grover_impact": "none", "estimated_qubits": 3500},
    "ECDH-P256": {"category": AlgorithmCategory.ECDH, "avs": 90, "shor_vulnerable": True, "grover_impact": "none", "estimated_qubits": 2500},
    "DH-2048": {"category": AlgorithmCategory.DH, "avs": 85, "shor_vulnerable": True, "grover_impact": "none", "estimated_qubits": 4000},
    # Symmetric — impacted by Grover (halved security)
    "AES-128": {"category": AlgorithmCategory.AES, "avs": 30, "shor_vulnerable": False, "grover_impact": "halved", "post_quantum_bits": 64},
    "AES-256": {"category": AlgorithmCategory.AES, "avs": 5, "shor_vulnerable": False, "grover_impact": "halved", "post_quantum_bits": 128},
    # Hash — impacted by Grover
    "SHA-256": {"category": AlgorithmCategory.SHA2, "avs": 15, "shor_vulnerable": False, "grover_impact": "halved"},
    "SHA3-256": {"category": AlgorithmCategory.SHA3, "avs": 5, "shor_vulnerable": False, "grover_impact": "halved"},
    # Post-Quantum — resistant
    "ML-KEM-768": {"category": AlgorithmCategory.ML_KEM, "avs": 2, "shor_vulnerable": False, "grover_impact": "none"},
    "ML-KEM-1024": {"category": AlgorithmCategory.ML_KEM, "avs": 1, "shor_vulnerable": False, "grover_impact": "none"},
    "ML-DSA-65": {"category": AlgorithmCategory.ML_DSA, "avs": 2, "shor_vulnerable": False, "grover_impact": "none"},
    "ML-DSA-87": {"category": AlgorithmCategory.ML_DSA, "avs": 1, "shor_vulnerable": False, "grover_impact": "none"},
    "Falcon-512": {"category": AlgorithmCategory.FALCON, "avs": 3, "shor_vulnerable": False, "grover_impact": "none"},
    "SLH-DSA-256f": {"category": AlgorithmCategory.SLH_DSA, "avs": 1, "shor_vulnerable": False, "grover_impact": "none"},
    "LMS-SHA256-H20": {"category": AlgorithmCategory.LMS, "avs": 1, "shor_vulnerable": False, "grover_impact": "none"},
    # Hybrid — transitional
    "X25519-ML-KEM-768": {"category": AlgorithmCategory.HYBRID, "avs": 5, "shor_vulnerable": False, "grover_impact": "none"},
}


@dataclass
class CryptoAsset:
    """A cryptographic asset to be assessed for quantum risk."""
    asset_id: str
    name: str
    algorithm: str                    # Algorithm name (key in ALGORITHM_VULNERABILITY)
    key_size_bits: int
    data_sensitivity: DataSensitivity
    data_retention_years: int         # How long data must remain confidential
    system_count: int = 1             # Number of systems using this asset
    migration_phase: MigrationPhase = MigrationPhase.NOT_STARTED
    domain: str = ""                  # banking, healthcare, automotive, etc.
    is_key_exchange: bool = False     # KEX algorithms face harvest-now risk
    last_rotation: Optional[float] = None


@dataclass
class ThreatAssessment:
    """
    Quantum threat assessment for a cryptographic asset.

    Contains the composite QRS score and individual factor breakdowns.
    """
    asset_id: str
    algorithm: str
    quantum_risk_score: float         # 0-100 composite score
    risk_level: RiskLevel
    algorithm_vulnerability: float    # AVS: 0-100
    time_horizon_risk: float          # THR: 0-100
    data_sensitivity_score: float     # DSS: 0-100
    migration_gap_score: float        # MGS: 0-100
    exposure_surface_score: float     # ES: 0-100
    harvest_now_risk: bool            # Is this asset at HNDL risk?
    recommended_action: str
    recommended_algorithm: str
    migration_deadline_year: int
    assessed_at: float = field(default_factory=time.time)


@dataclass
class PortfolioAssessment:
    """Assessment of an entire cryptographic asset portfolio."""
    total_assets: int
    critical_count: int
    high_count: int
    moderate_count: int
    low_count: int
    average_qrs: float
    most_vulnerable: List[ThreatAssessment]
    harvest_now_at_risk: int
    migration_coverage: float          # % of assets at PQC or hybrid
    cnsa2_readiness: float             # % meeting CNSA 2.0
    assessments: List[ThreatAssessment]
    assessed_at: float = field(default_factory=time.time)


class QuantumThreatScorer:
    """
    AI-driven quantum threat scoring engine.

    Evaluates cryptographic assets and produces risk scores
    calibrated against quantum computing progress timelines.

    Usage:
        scorer = QuantumThreatScorer()

        # Assess individual asset
        asset = CryptoAsset(
            asset_id="wire-transfer-signing",
            algorithm="ECDSA-P256",
            key_size_bits=256,
            data_sensitivity=DataSensitivity.SECRET,
            data_retention_years=7,
            system_count=150,
        )
        assessment = scorer.assess_asset(asset)

        # Assess entire portfolio
        portfolio = scorer.assess_portfolio([asset1, asset2, ...])
    """

    def __init__(
        self,
        quantum_timeline_year: int = 2035,
        current_year: int = 2026,
        cnsa2_deadline_year: int = 2035,
    ):
        """
        Args:
            quantum_timeline_year: Estimated year a CRQC can break RSA-2048
            current_year: Current year for timeline calculations
            cnsa2_deadline_year: CNSA 2.0 compliance deadline
        """
        self.quantum_timeline_year = quantum_timeline_year
        self.current_year = current_year
        self.cnsa2_deadline_year = cnsa2_deadline_year

        # Scoring weights (sum to 1.0)
        self._weights = {
            "algorithm_vulnerability": 0.30,
            "time_horizon": 0.25,
            "data_sensitivity": 0.20,
            "migration_gap": 0.15,
            "exposure_surface": 0.10,
        }

        logger.info(
            f"Quantum threat scorer: timeline={quantum_timeline_year}, "
            f"cnsa2_deadline={cnsa2_deadline_year}"
        )

    def assess_asset(self, asset: CryptoAsset) -> ThreatAssessment:
        """
        Assess a single cryptographic asset for quantum risk.

        Returns a comprehensive threat assessment with composite
        Quantum Risk Score (QRS) and factor breakdown.
        """
        # 1. Algorithm Vulnerability Score (AVS)
        avs = self._compute_avs(asset)

        # 2. Time Horizon Risk (THR)
        thr = self._compute_thr(asset)

        # 3. Data Sensitivity Score (DSS)
        dss = self._compute_dss(asset)

        # 4. Migration Gap Score (MGS)
        mgs = self._compute_mgs(asset)

        # 5. Exposure Surface (ES)
        es = self._compute_es(asset)

        # Composite QRS
        qrs = (
            self._weights["algorithm_vulnerability"] * avs
            + self._weights["time_horizon"] * thr
            + self._weights["data_sensitivity"] * dss
            + self._weights["migration_gap"] * mgs
            + self._weights["exposure_surface"] * es
        )

        qrs = min(100, max(0, qrs))

        # Determine risk level
        if qrs >= 76:
            risk_level = RiskLevel.CRITICAL
        elif qrs >= 51:
            risk_level = RiskLevel.HIGH
        elif qrs >= 26:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW

        # Harvest-now-decrypt-later risk
        algo_info = ALGORITHM_VULNERABILITY.get(asset.algorithm, {})
        harvest_risk = (
            algo_info.get("shor_vulnerable", False)
            and asset.data_retention_years > (self.quantum_timeline_year - self.current_year)
        )

        # Recommendations
        action, target_algo, deadline = self._generate_recommendation(asset, qrs, risk_level)

        assessment = ThreatAssessment(
            asset_id=asset.asset_id,
            algorithm=asset.algorithm,
            quantum_risk_score=round(qrs, 1),
            risk_level=risk_level,
            algorithm_vulnerability=round(avs, 1),
            time_horizon_risk=round(thr, 1),
            data_sensitivity_score=round(dss, 1),
            migration_gap_score=round(mgs, 1),
            exposure_surface_score=round(es, 1),
            harvest_now_risk=harvest_risk,
            recommended_action=action,
            recommended_algorithm=target_algo,
            migration_deadline_year=deadline,
        )

        QTS_SCORE.observe(qrs)
        QTS_ASSESSMENTS.inc()

        if risk_level == RiskLevel.CRITICAL:
            QTS_CRITICAL_ASSETS.inc()

        return assessment

    def assess_portfolio(self, assets: List[CryptoAsset]) -> PortfolioAssessment:
        """Assess an entire portfolio of cryptographic assets."""
        assessments = [self.assess_asset(a) for a in assets]

        critical = sum(1 for a in assessments if a.risk_level == RiskLevel.CRITICAL)
        high = sum(1 for a in assessments if a.risk_level == RiskLevel.HIGH)
        moderate = sum(1 for a in assessments if a.risk_level == RiskLevel.MODERATE)
        low = sum(1 for a in assessments if a.risk_level == RiskLevel.LOW)

        avg_qrs = sum(a.quantum_risk_score for a in assessments) / len(assessments) if assessments else 0

        # Sort by QRS descending for most vulnerable
        sorted_assessments = sorted(assessments, key=lambda a: a.quantum_risk_score, reverse=True)

        # Migration coverage
        pqc_or_hybrid = sum(
            1 for a in assets
            if a.migration_phase in (
                MigrationPhase.HYBRID_DEPLOYMENT,
                MigrationPhase.PQC_PRIMARY,
                MigrationPhase.PQC_ONLY,
                MigrationPhase.CNSA2_COMPLIANT,
            )
        )
        migration_coverage = pqc_or_hybrid / len(assets) * 100 if assets else 0

        cnsa2_ready = sum(
            1 for a in assets
            if a.migration_phase == MigrationPhase.CNSA2_COMPLIANT
        )
        cnsa2_readiness = cnsa2_ready / len(assets) * 100 if assets else 0

        harvest_at_risk = sum(1 for a in assessments if a.harvest_now_risk)

        portfolio = PortfolioAssessment(
            total_assets=len(assets),
            critical_count=critical,
            high_count=high,
            moderate_count=moderate,
            low_count=low,
            average_qrs=round(avg_qrs, 1),
            most_vulnerable=sorted_assessments[:10],
            harvest_now_at_risk=harvest_at_risk,
            migration_coverage=round(migration_coverage, 1),
            cnsa2_readiness=round(cnsa2_readiness, 1),
            assessments=assessments,
        )

        logger.info(
            f"Portfolio assessed: {len(assets)} assets, "
            f"avg_QRS={avg_qrs:.1f}, critical={critical}, "
            f"migration={migration_coverage:.0f}%"
        )

        return portfolio

    # ── Factor computation ────────────────────────────────────────

    def _compute_avs(self, asset: CryptoAsset) -> float:
        """Compute Algorithm Vulnerability Score."""
        algo_info = ALGORITHM_VULNERABILITY.get(asset.algorithm)
        if algo_info:
            return algo_info["avs"]

        # Unknown algorithm — assume moderate risk
        return 50.0

    def _compute_thr(self, asset: CryptoAsset) -> float:
        """
        Compute Time Horizon Risk using Mosca's inequality.

        Mosca's theorem: If x + y > z, migration is urgent.
        x = time data needs protection (retention years)
        y = time to migrate to PQC
        z = time until quantum computer breaks algorithm

        Higher THR = more urgent migration needed.
        """
        algo_info = ALGORITHM_VULNERABILITY.get(asset.algorithm, {})

        if not algo_info.get("shor_vulnerable", False):
            return 5.0  # PQC algorithms have low time horizon risk

        z = self.quantum_timeline_year - self.current_year  # Years until threat
        x = asset.data_retention_years                       # Years of protection needed

        # Estimate migration time based on current phase
        migration_times = {
            MigrationPhase.NOT_STARTED: 5,
            MigrationPhase.PLANNING: 4,
            MigrationPhase.TESTING: 3,
            MigrationPhase.HYBRID_DEPLOYMENT: 1,
            MigrationPhase.PQC_PRIMARY: 0.5,
            MigrationPhase.PQC_ONLY: 0,
            MigrationPhase.CNSA2_COMPLIANT: 0,
        }
        y = migration_times.get(asset.migration_phase, 5)

        # Mosca urgency: how much x + y exceeds z
        mosca_gap = (x + y) - z

        if mosca_gap > 5:
            return 100.0
        elif mosca_gap > 0:
            return 60 + (mosca_gap / 5) * 40
        else:
            # Still have time, but closer = higher risk
            margin = abs(mosca_gap)
            return max(0, 60 - margin * 10)

    def _compute_dss(self, asset: CryptoAsset) -> float:
        """Compute Data Sensitivity Score."""
        sensitivity_scores = {
            DataSensitivity.PUBLIC: 10,
            DataSensitivity.INTERNAL: 30,
            DataSensitivity.CONFIDENTIAL: 60,
            DataSensitivity.SECRET: 85,
            DataSensitivity.TOP_SECRET: 100,
        }
        return sensitivity_scores.get(asset.data_sensitivity, 50)

    def _compute_mgs(self, asset: CryptoAsset) -> float:
        """Compute Migration Gap Score."""
        phase_scores = {
            MigrationPhase.NOT_STARTED: 100,
            MigrationPhase.PLANNING: 80,
            MigrationPhase.TESTING: 60,
            MigrationPhase.HYBRID_DEPLOYMENT: 30,
            MigrationPhase.PQC_PRIMARY: 10,
            MigrationPhase.PQC_ONLY: 2,
            MigrationPhase.CNSA2_COMPLIANT: 0,
        }
        return phase_scores.get(asset.migration_phase, 100)

    def _compute_es(self, asset: CryptoAsset) -> float:
        """Compute Exposure Surface score based on system count."""
        if asset.system_count <= 1:
            return 10
        elif asset.system_count <= 10:
            return 30
        elif asset.system_count <= 100:
            return 60
        elif asset.system_count <= 1000:
            return 80
        else:
            return 100

    def _generate_recommendation(
        self,
        asset: CryptoAsset,
        qrs: float,
        risk_level: RiskLevel,
    ) -> Tuple[str, str, int]:
        """Generate migration recommendation."""
        algo_info = ALGORITHM_VULNERABILITY.get(asset.algorithm, {})
        category = algo_info.get("category")

        # Target algorithm recommendation
        if category in (AlgorithmCategory.RSA, AlgorithmCategory.DH):
            if asset.is_key_exchange:
                target = "ML-KEM-768"
            else:
                target = "ML-DSA-65"
        elif category in (AlgorithmCategory.ECDSA, AlgorithmCategory.ECDH):
            if asset.is_key_exchange:
                target = "X25519-ML-KEM-768"
            else:
                target = "ML-DSA-65"
        elif category == AlgorithmCategory.AES:
            target = "AES-256" if asset.algorithm != "AES-256" else asset.algorithm
        elif category in (AlgorithmCategory.ML_KEM, AlgorithmCategory.ML_DSA,
                         AlgorithmCategory.FALCON, AlgorithmCategory.SLH_DSA,
                         AlgorithmCategory.LMS, AlgorithmCategory.XMSS):
            target = asset.algorithm  # Already PQC
        else:
            target = "ML-KEM-768"

        # Action and deadline
        if risk_level == RiskLevel.CRITICAL:
            action = "IMMEDIATE: Begin emergency PQC migration. Deploy hybrid mode within 90 days."
            deadline = self.current_year
        elif risk_level == RiskLevel.HIGH:
            action = "URGENT: Initiate PQC migration plan. Deploy hybrid mode within 6 months."
            deadline = self.current_year + 1
        elif risk_level == RiskLevel.MODERATE:
            action = "PLAN: Include in next migration cycle. Target hybrid deployment within 18 months."
            deadline = self.current_year + 2
        else:
            action = "MONITOR: Low risk. Review at next annual assessment."
            deadline = min(self.cnsa2_deadline_year, self.current_year + 5)

        return action, target, deadline
