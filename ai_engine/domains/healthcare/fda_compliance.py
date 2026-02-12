"""
FDA Cybersecurity Compliance Module

Supports FDA premarket cybersecurity guidance (2023/2025):
- Cryptographic agility requirements
- Quantum-safe transition planning
- Security risk management
- Vulnerability disclosure
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CybersecurityRequirement(Enum):
    """FDA cybersecurity requirements categories."""

    CRYPTO_AGILITY = auto()
    AUTHENTICATION = auto()
    AUTHORIZATION = auto()
    DATA_PROTECTION = auto()
    INTEGRITY = auto()
    AVAILABILITY = auto()
    UPDATE_CAPABILITY = auto()
    MONITORING = auto()
    INCIDENT_RESPONSE = auto()


@dataclass
class ComplianceEvidence:
    """Evidence supporting compliance with a requirement."""

    requirement: CybersecurityRequirement
    description: str
    evidence_type: str  # "design", "test", "documentation"
    artifacts: List[str] = field(default_factory=list)
    verified: bool = False
    verified_date: Optional[datetime] = None


@dataclass
class PremarketSubmission:
    """FDA premarket cybersecurity submission."""

    submission_id: str
    device_name: str
    device_class: str  # "I", "II", or "III"
    submission_type: str  # "510k", "PMA", "De Novo"
    pqc_transition_plan: Optional[str] = None
    crypto_agility_evidence: List[ComplianceEvidence] = field(default_factory=list)


class FDAComplianceValidator:
    """
    Validates device compliance with FDA cybersecurity guidance.

    Key areas for PQC:
    1. Cryptographic agility - ability to update algorithms
    2. Quantum-safe transition plan
    3. Key management lifecycle
    4. Secure update mechanisms
    """

    def __init__(self, device_id: str):
        self.device_id = device_id
        self._evidence: List[ComplianceEvidence] = []

    def add_evidence(self, evidence: ComplianceEvidence) -> None:
        """Add compliance evidence."""
        self._evidence.append(evidence)
        logger.info(f"Added evidence for {evidence.requirement.name}")

    def validate_crypto_agility(self) -> Dict[str, Any]:
        """Validate cryptographic agility requirements."""
        checks = {
            "algorithm_update_capability": self._check_algorithm_update(),
            "key_rotation_support": self._check_key_rotation(),
            "pqc_readiness": self._check_pqc_readiness(),
            "hybrid_mode_support": self._check_hybrid_support(),
        }

        passed = all(checks.values())

        return {
            "requirement": "CRYPTO_AGILITY",
            "passed": passed,
            "checks": checks,
            "recommendation": self._get_recommendation(checks),
        }

    def _check_algorithm_update(self) -> bool:
        """Check if device supports algorithm updates."""
        evidence = [
            e
            for e in self._evidence
            if e.requirement == CybersecurityRequirement.CRYPTO_AGILITY and "algorithm_update" in e.description.lower()
        ]
        return len(evidence) > 0 and any(e.verified for e in evidence)

    def _check_key_rotation(self) -> bool:
        """Check if device supports key rotation."""
        evidence = [
            e
            for e in self._evidence
            if e.requirement == CybersecurityRequirement.CRYPTO_AGILITY and "key_rotation" in e.description.lower()
        ]
        return len(evidence) > 0

    def _check_pqc_readiness(self) -> bool:
        """Check if device is PQC-ready."""
        evidence = [e for e in self._evidence if "pqc" in e.description.lower() or "quantum" in e.description.lower()]
        return len(evidence) > 0

    def _check_hybrid_support(self) -> bool:
        """Check if device supports hybrid classical/PQC."""
        evidence = [e for e in self._evidence if "hybrid" in e.description.lower()]
        return len(evidence) > 0

    def _get_recommendation(self, checks: Dict[str, bool]) -> str:
        """Get recommendation based on check results."""
        failed = [k for k, v in checks.items() if not v]
        if not failed:
            return "Device meets FDA cryptographic agility requirements"

        return f"Address the following gaps: {', '.join(failed)}"

    def generate_submission_report(self) -> Dict[str, Any]:
        """Generate FDA submission report."""
        crypto_agility = self.validate_crypto_agility()

        return {
            "device_id": self.device_id,
            "report_date": datetime.now().isoformat(),
            "sections": {
                "cryptographic_agility": crypto_agility,
            },
            "evidence_count": len(self._evidence),
            "overall_status": "PASS" if crypto_agility["passed"] else "NEEDS_WORK",
        }
