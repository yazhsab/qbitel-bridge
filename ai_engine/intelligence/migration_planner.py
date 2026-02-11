"""
Migration Planner

AI-powered migration planning from legacy to modern systems.
Creates detailed migration plans with phases, risks, and recommendations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


logger = logging.getLogger(__name__)


class MigrationStrategy(Enum):
    """Migration strategies."""

    REPLATFORM = "replatform"  # Lift and shift with minimal changes
    REFACTOR = "refactor"  # Restructure for cloud-native
    REARCHITECT = "rearchitect"  # Complete redesign
    REPLACE = "replace"  # Replace with COTS/SaaS
    ENCAPSULATE = "encapsulate"  # Wrap with modern APIs
    HYBRID = "hybrid"  # Combination approach


class MigrationRisk(Enum):
    """Migration risk levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MigrationPhaseType(Enum):
    """Types of migration phases."""

    ASSESSMENT = "assessment"
    DESIGN = "design"
    BUILD = "build"
    TEST = "test"
    MIGRATE = "migrate"
    VALIDATE = "validate"
    CUTOVER = "cutover"
    DECOMMISSION = "decommission"


@dataclass
class MigrationTask:
    """Individual migration task."""

    task_id: str
    name: str
    description: str
    phase: MigrationPhaseType
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    skills_required: List[str] = field(default_factory=list)
    risk_level: MigrationRisk = MigrationRisk.MEDIUM
    automated: bool = False
    automation_tool: Optional[str] = None


@dataclass
class MigrationPhase:
    """Migration phase definition."""

    phase_id: str
    name: str
    phase_type: MigrationPhaseType
    description: str
    tasks: List[MigrationTask] = field(default_factory=list)
    entry_criteria: List[str] = field(default_factory=list)
    exit_criteria: List[str] = field(default_factory=list)
    risks: List[Dict[str, Any]] = field(default_factory=list)
    order: int = 0


@dataclass
class DataMigrationSpec:
    """Data migration specification."""

    source_format: str
    target_format: str
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    volume_estimate: str = ""
    rollback_strategy: str = ""


@dataclass
class SecurityMigrationSpec:
    """Security migration specification."""

    current_crypto: List[str] = field(default_factory=list)
    target_crypto: List[str] = field(default_factory=list)
    pqc_requirements: bool = True
    key_migration_plan: str = ""
    certificate_migration: str = ""
    hsm_migration: str = ""


@dataclass
class MigrationPlan:
    """Complete migration plan."""

    plan_id: str
    name: str
    strategy: MigrationStrategy
    source_system: str
    target_system: str

    # Phases
    phases: List[MigrationPhase] = field(default_factory=list)

    # Specifications
    data_migration: Optional[DataMigrationSpec] = None
    security_migration: Optional[SecurityMigrationSpec] = None

    # Risks and mitigations
    risks: List[Dict[str, Any]] = field(default_factory=list)
    mitigations: List[Dict[str, Any]] = field(default_factory=list)

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    complexity_score: float = 0.0
    pqc_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "strategy": self.strategy.value,
            "source_system": self.source_system,
            "target_system": self.target_system,
            "phases": [
                {
                    "phase_id": p.phase_id,
                    "name": p.name,
                    "type": p.phase_type.value,
                    "tasks_count": len(p.tasks),
                    "risks_count": len(p.risks),
                }
                for p in self.phases
            ],
            "total_tasks": sum(len(p.tasks) for p in self.phases),
            "risks_count": len(self.risks),
            "complexity_score": self.complexity_score,
            "pqc_ready": self.pqc_ready,
            "created_at": self.created_at.isoformat(),
        }


class MigrationPlanner:
    """
    AI-powered migration planner.

    Creates comprehensive migration plans based on:
    - Source system analysis
    - Target architecture requirements
    - Security requirements (including PQC)
    - Risk assessment
    """

    def __init__(self):
        self._strategy_templates = self._build_strategy_templates()

    def create_plan(
        self,
        source_analysis: Dict[str, Any],
        target_requirements: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> MigrationPlan:
        """
        Create a migration plan.

        Args:
            source_analysis: Analysis of source system (from ProtocolAnalyzer/Fingerprinter)
            target_requirements: Target system requirements
            constraints: Constraints (time, budget, resources)

        Returns:
            MigrationPlan with detailed phases and tasks
        """
        # Determine best strategy
        strategy = self._determine_strategy(source_analysis, target_requirements)

        # Create plan
        plan = MigrationPlan(
            plan_id=str(uuid.uuid4()),
            name=f"Migration from {source_analysis.get('system_type', 'Legacy')} to {target_requirements.get('target_platform', 'Modern')}",
            strategy=strategy,
            source_system=source_analysis.get("system_type", "unknown"),
            target_system=target_requirements.get("target_platform", "cloud"),
        )

        # Build phases
        plan.phases = self._build_phases(strategy, source_analysis, target_requirements)

        # Create data migration spec
        plan.data_migration = self._create_data_migration_spec(source_analysis, target_requirements)

        # Create security migration spec
        plan.security_migration = self._create_security_migration_spec(
            source_analysis, target_requirements
        )

        # Assess risks
        plan.risks = self._assess_risks(source_analysis, strategy)

        # Create mitigations
        plan.mitigations = self._create_mitigations(plan.risks)

        # Calculate complexity
        plan.complexity_score = self._calculate_complexity(plan)

        # Check PQC readiness
        plan.pqc_ready = self._check_pqc_readiness(plan)

        return plan

    def create_pqc_migration_plan(
        self,
        current_crypto: Dict[str, Any],
        target_security_level: str = "level3",
    ) -> MigrationPlan:
        """
        Create a PQC migration plan.

        Args:
            current_crypto: Current cryptographic algorithms and keys
            target_security_level: Target PQC security level (level1, level3, level5)

        Returns:
            MigrationPlan for PQC migration
        """
        plan = MigrationPlan(
            plan_id=str(uuid.uuid4()),
            name="Post-Quantum Cryptography Migration",
            strategy=MigrationStrategy.HYBRID,
            source_system="classical_crypto",
            target_system="pqc_hybrid",
            pqc_ready=True,
        )

        # Assessment phase
        assessment_phase = MigrationPhase(
            phase_id=str(uuid.uuid4()),
            name="Cryptographic Inventory",
            phase_type=MigrationPhaseType.ASSESSMENT,
            description="Inventory all cryptographic assets and usage",
            order=1,
            tasks=[
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Key Inventory",
                    description="Inventory all cryptographic keys and their usage",
                    phase=MigrationPhaseType.ASSESSMENT,
                    automated=True,
                    automation_tool="Qbitel AI Key Discovery",
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Algorithm Analysis",
                    description="Analyze algorithms for quantum vulnerability",
                    phase=MigrationPhaseType.ASSESSMENT,
                    automated=True,
                    automation_tool="Qbitel AI Protocol Analyzer",
                ),
            ],
        )

        # Build phase
        build_phase = MigrationPhase(
            phase_id=str(uuid.uuid4()),
            name="PQC Implementation",
            phase_type=MigrationPhaseType.BUILD,
            description="Implement PQC algorithms",
            order=2,
            tasks=[
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Deploy ML-KEM",
                    description="Deploy ML-KEM-768 for key encapsulation",
                    phase=MigrationPhaseType.BUILD,
                    skills_required=["PQC", "Cryptography", "HSM"],
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Deploy ML-DSA",
                    description="Deploy ML-DSA-65 for digital signatures",
                    phase=MigrationPhaseType.BUILD,
                    skills_required=["PQC", "Cryptography", "PKI"],
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Hybrid Mode Configuration",
                    description="Configure hybrid classical+PQC mode",
                    phase=MigrationPhaseType.BUILD,
                    skills_required=["Cryptography"],
                ),
            ],
        )

        # Migration phase
        migrate_phase = MigrationPhase(
            phase_id=str(uuid.uuid4()),
            name="Key Migration",
            phase_type=MigrationPhaseType.MIGRATE,
            description="Migrate keys to PQC-protected storage",
            order=3,
            tasks=[
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="KEK Migration",
                    description="Migrate Key Encryption Keys to PQC",
                    phase=MigrationPhaseType.MIGRATE,
                    risk_level=MigrationRisk.HIGH,
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Certificate Reissuance",
                    description="Reissue certificates with PQC signatures",
                    phase=MigrationPhaseType.MIGRATE,
                    risk_level=MigrationRisk.MEDIUM,
                ),
            ],
        )

        plan.phases = [assessment_phase, build_phase, migrate_phase]

        # Security spec
        plan.security_migration = SecurityMigrationSpec(
            current_crypto=list(current_crypto.keys()),
            target_crypto=["ML-KEM-768", "ML-DSA-65", "AES-256-GCM"],
            pqc_requirements=True,
            key_migration_plan="Hybrid wrap with ML-KEM during transition",
            hsm_migration="Update HSM firmware for PQC support",
        )

        plan.risks = [
            {
                "name": "Performance Impact",
                "level": MigrationRisk.MEDIUM.value,
                "description": "PQC operations may have higher latency",
            },
            {
                "name": "Interoperability",
                "level": MigrationRisk.HIGH.value,
                "description": "Partners may not support PQC yet",
            },
        ]

        return plan

    def _determine_strategy(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> MigrationStrategy:
        """Determine best migration strategy."""
        source_type = source.get("system_type", "").lower()
        target_platform = target.get("target_platform", "").lower()

        # Check for specific scenarios
        if "cloud" in target_platform:
            if source.get("modernization_complexity") == "low":
                return MigrationStrategy.REPLATFORM
            elif source.get("modernization_complexity") == "very_high":
                return MigrationStrategy.ENCAPSULATE
            else:
                return MigrationStrategy.REFACTOR

        if target.get("replace_with_cots"):
            return MigrationStrategy.REPLACE

        if "microservices" in target_platform:
            return MigrationStrategy.REARCHITECT

        return MigrationStrategy.REFACTOR

    def _build_phases(
        self,
        strategy: MigrationStrategy,
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> List[MigrationPhase]:
        """Build migration phases based on strategy."""
        phases = []

        # Assessment phase (always first)
        phases.append(MigrationPhase(
            phase_id=str(uuid.uuid4()),
            name="Assessment & Discovery",
            phase_type=MigrationPhaseType.ASSESSMENT,
            description="Complete analysis of source system and requirements",
            order=1,
            entry_criteria=["Project kickoff complete", "Stakeholders identified"],
            exit_criteria=["Source system fully documented", "Requirements validated"],
            tasks=[
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Protocol Analysis",
                    description="Analyze all protocols and data formats",
                    phase=MigrationPhaseType.ASSESSMENT,
                    automated=True,
                    automation_tool="Qbitel AI Protocol Analyzer",
                    deliverables=["Protocol inventory", "Data dictionary"],
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Legacy Fingerprinting",
                    description="Fingerprint legacy system components",
                    phase=MigrationPhaseType.ASSESSMENT,
                    automated=True,
                    automation_tool="Qbitel AI Fingerprinter",
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Security Assessment",
                    description="Assess current security posture",
                    phase=MigrationPhaseType.ASSESSMENT,
                    automated=True,
                    automation_tool="Qbitel AI Security Assessor",
                ),
            ],
        ))

        # Design phase
        phases.append(MigrationPhase(
            phase_id=str(uuid.uuid4()),
            name="Architecture Design",
            phase_type=MigrationPhaseType.DESIGN,
            description="Design target architecture and migration approach",
            order=2,
            entry_criteria=["Assessment complete"],
            exit_criteria=["Architecture approved", "Migration approach defined"],
            tasks=[
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Target Architecture",
                    description="Design target state architecture",
                    phase=MigrationPhaseType.DESIGN,
                    skills_required=["Enterprise Architecture", "Cloud"],
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Data Mapping",
                    description="Create field-level data mappings",
                    phase=MigrationPhaseType.DESIGN,
                    automated=True,
                    automation_tool="Qbitel AI Protocol Mapper",
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Security Architecture",
                    description="Design PQC-ready security architecture",
                    phase=MigrationPhaseType.DESIGN,
                    skills_required=["Security Architecture", "PQC"],
                ),
            ],
        ))

        # Build phase
        phases.append(MigrationPhase(
            phase_id=str(uuid.uuid4()),
            name="Build & Configure",
            phase_type=MigrationPhaseType.BUILD,
            description="Build target components and configure integration",
            order=3,
            tasks=[
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Protocol Adapters",
                    description="Build protocol adapters and transformers",
                    phase=MigrationPhaseType.BUILD,
                    automated=True,
                    automation_tool="Qbitel AI Zero-Touch Builder",
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Security Implementation",
                    description="Implement PQC and HSM integration",
                    phase=MigrationPhaseType.BUILD,
                    skills_required=["Cryptography", "HSM"],
                ),
            ],
        ))

        # Test phase
        phases.append(MigrationPhase(
            phase_id=str(uuid.uuid4()),
            name="Testing & Validation",
            phase_type=MigrationPhaseType.TEST,
            description="Comprehensive testing of migrated components",
            order=4,
            tasks=[
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Protocol Validation",
                    description="Validate protocol transformations",
                    phase=MigrationPhaseType.TEST,
                    automated=True,
                    automation_tool="Qbitel AI Validator",
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Security Testing",
                    description="Penetration testing and security validation",
                    phase=MigrationPhaseType.TEST,
                ),
            ],
        ))

        # Migration phase
        phases.append(MigrationPhase(
            phase_id=str(uuid.uuid4()),
            name="Data Migration",
            phase_type=MigrationPhaseType.MIGRATE,
            description="Execute data and workload migration",
            order=5,
            risks=[
                {"name": "Data loss", "level": "high"},
                {"name": "Downtime", "level": "medium"},
            ],
            tasks=[
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Data Extraction",
                    description="Extract data from source system",
                    phase=MigrationPhaseType.MIGRATE,
                    risk_level=MigrationRisk.HIGH,
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Data Transformation",
                    description="Transform data to target format",
                    phase=MigrationPhaseType.MIGRATE,
                    automated=True,
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Data Load",
                    description="Load data into target system",
                    phase=MigrationPhaseType.MIGRATE,
                ),
            ],
        ))

        # Cutover phase
        phases.append(MigrationPhase(
            phase_id=str(uuid.uuid4()),
            name="Cutover",
            phase_type=MigrationPhaseType.CUTOVER,
            description="Production cutover to new system",
            order=6,
            risks=[
                {"name": "Business disruption", "level": "critical"},
            ],
            tasks=[
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Traffic Routing",
                    description="Route traffic to new system",
                    phase=MigrationPhaseType.CUTOVER,
                    risk_level=MigrationRisk.CRITICAL,
                ),
                MigrationTask(
                    task_id=str(uuid.uuid4()),
                    name="Monitoring Setup",
                    description="Configure production monitoring",
                    phase=MigrationPhaseType.CUTOVER,
                ),
            ],
        ))

        return phases

    def _create_data_migration_spec(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> DataMigrationSpec:
        """Create data migration specification."""
        return DataMigrationSpec(
            source_format=source.get("protocol_type", "unknown"),
            target_format=target.get("target_format", "ISO20022"),
            transformations=[
                {
                    "type": "encoding",
                    "from": source.get("encoding", "EBCDIC"),
                    "to": "UTF-8",
                },
                {
                    "type": "date_format",
                    "from": "YYMMDD",
                    "to": "ISO8601",
                },
            ],
            validation_rules=[
                "Schema validation against target XSD",
                "Business rule validation",
                "Referential integrity checks",
            ],
            rollback_strategy="Point-in-time recovery from pre-migration snapshot",
        )

    def _create_security_migration_spec(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> SecurityMigrationSpec:
        """Create security migration specification."""
        current_crypto = []
        if source.get("encryption_detected"):
            current_crypto.append("DES/3DES")
        if source.get("signature_detected"):
            current_crypto.append("RSA-2048")

        return SecurityMigrationSpec(
            current_crypto=current_crypto or ["Unknown"],
            target_crypto=["ML-KEM-768", "ML-DSA-65", "AES-256-GCM"],
            pqc_requirements=True,
            key_migration_plan="Re-encrypt keys under PQC-protected KEK",
            certificate_migration="Issue new certificates with hybrid signatures",
            hsm_migration="Migrate to cloud HSM with PQC support",
        )

    def _assess_risks(
        self,
        source: Dict[str, Any],
        strategy: MigrationStrategy,
    ) -> List[Dict[str, Any]]:
        """Assess migration risks."""
        risks = []

        # Encoding risk
        if source.get("encoding") == "EBCDIC":
            risks.append({
                "id": str(uuid.uuid4()),
                "name": "Character Encoding",
                "level": MigrationRisk.MEDIUM.value,
                "description": "EBCDIC to UTF-8 conversion may cause data corruption",
                "probability": "medium",
                "impact": "high",
            })

        # Complexity risk
        if source.get("modernization_complexity") in ("high", "very_high"):
            risks.append({
                "id": str(uuid.uuid4()),
                "name": "System Complexity",
                "level": MigrationRisk.HIGH.value,
                "description": "High system complexity increases migration risk",
                "probability": "high",
                "impact": "high",
            })

        # Strategy-specific risks
        if strategy == MigrationStrategy.REARCHITECT:
            risks.append({
                "id": str(uuid.uuid4()),
                "name": "Scope Creep",
                "level": MigrationRisk.HIGH.value,
                "description": "Re-architecture may lead to scope expansion",
                "probability": "high",
                "impact": "medium",
            })

        # Security risks
        risks.append({
            "id": str(uuid.uuid4()),
            "name": "Key Material Exposure",
            "level": MigrationRisk.CRITICAL.value,
            "description": "Cryptographic keys exposed during migration",
            "probability": "low",
            "impact": "critical",
        })

        return risks

    def _create_mitigations(
        self,
        risks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create risk mitigations."""
        mitigations = []

        for risk in risks:
            mitigation = {
                "risk_id": risk["id"],
                "risk_name": risk["name"],
                "mitigations": [],
            }

            if "encoding" in risk["name"].lower():
                mitigation["mitigations"].extend([
                    "Use validated encoding conversion libraries",
                    "Implement comprehensive character set testing",
                    "Maintain rollback capability",
                ])

            if "complexity" in risk["name"].lower():
                mitigation["mitigations"].extend([
                    "Break migration into smaller increments",
                    "Use Qbitel AI for automated analysis",
                    "Implement continuous validation",
                ])

            if "key" in risk["name"].lower():
                mitigation["mitigations"].extend([
                    "Use HSM-to-HSM secure key transfer",
                    "Implement PQC hybrid key wrapping",
                    "Maintain audit trail of all key operations",
                ])

            mitigations.append(mitigation)

        return mitigations

    def _calculate_complexity(self, plan: MigrationPlan) -> float:
        """Calculate migration complexity score."""
        score = 0.0

        # Task count factor
        total_tasks = sum(len(p.tasks) for p in plan.phases)
        score += min(total_tasks / 50, 1.0) * 0.3

        # Risk factor
        critical_risks = sum(1 for r in plan.risks if r.get("level") == "critical")
        high_risks = sum(1 for r in plan.risks if r.get("level") == "high")
        score += min((critical_risks * 0.3 + high_risks * 0.15), 1.0) * 0.4

        # Security migration factor
        if plan.security_migration and plan.security_migration.pqc_requirements:
            score += 0.15

        # Data migration factor
        if plan.data_migration:
            score += len(plan.data_migration.transformations) * 0.02

        return min(score, 1.0)

    def _check_pqc_readiness(self, plan: MigrationPlan) -> bool:
        """Check if plan is PQC ready."""
        if not plan.security_migration:
            return False

        pqc_algos = ["ML-KEM", "ML-DSA", "Kyber", "Dilithium"]
        target_crypto = plan.security_migration.target_crypto

        return any(
            any(pqc in crypto for pqc in pqc_algos)
            for crypto in target_crypto
        )

    def _build_strategy_templates(self) -> Dict[MigrationStrategy, Dict[str, Any]]:
        """Build templates for each migration strategy."""
        return {
            MigrationStrategy.REPLATFORM: {
                "description": "Migrate with minimal changes",
                "typical_duration": "3-6 months",
                "risk_level": "low",
            },
            MigrationStrategy.REFACTOR: {
                "description": "Restructure for modern architecture",
                "typical_duration": "6-12 months",
                "risk_level": "medium",
            },
            MigrationStrategy.REARCHITECT: {
                "description": "Complete redesign for cloud-native",
                "typical_duration": "12-24 months",
                "risk_level": "high",
            },
        }
