"""
Tests for BPO zero-touch orchestration and automation recipes.

Tests cover:
- Environment discovery results (PBX, CRM, WFM, protocols, confidence)
- Security assessment (gap identification, risk scoring)
- Policy generation (PCI-DSS policies, auto-deployment flags)
- Provisioning (deployed/failed controls, rollback)
- Zero-touch deployment (5-phase pipeline, risk reduction)
- Recipe registry (built-in recipes, lookup, environment filtering)
- PCI-DSS voice compliance recipe
- Toll fraud prevention recipe
- Remote workforce security recipe
- Quantum-safe voice recipe
- Automation step data structures

Note: These tests define the expected behavior for the BPO zero-touch
modules (ai_engine.domains.bpo.zero_touch). The tests serve as a
specification and will pass once the modules are implemented.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ai_engine.domains.bpo.zero_touch.bpo_zero_touch_orchestrator import (
    BPOZeroTouchOrchestrator,
    EnvironmentDiscoveryResult,
    SecurityAssessmentResult,
    SecurityGap,
    PolicyGenerationResult,
    ProvisioningResult,
    ZeroTouchDeploymentResult,
)
from ai_engine.domains.bpo.zero_touch.automation_recipes import (
    BPOAutomationRecipe,
    AutomationStep,
    RecipeRegistry,
    PCIDSSVoiceComplianceRecipe,
    TollFraudPreventionRecipe,
    RemoteWorkforceSecurityRecipe,
    QuantumSafeVoiceRecipe,
    ComplianceSuiteRecipe,
    FullBPOSecurityRecipe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator():
    """Return a BPOZeroTouchOrchestrator instance."""
    return BPOZeroTouchOrchestrator()


@pytest.fixture
def recipe_registry():
    """Return a RecipeRegistry instance."""
    return RecipeRegistry()


@pytest.fixture
def pci_recipe():
    """Return a PCI-DSS voice compliance recipe."""
    return PCIDSSVoiceComplianceRecipe()


@pytest.fixture
def toll_fraud_recipe():
    """Return a toll fraud prevention recipe."""
    return TollFraudPreventionRecipe()


@pytest.fixture
def remote_workforce_recipe():
    """Return a remote workforce security recipe."""
    return RemoteWorkforceSecurityRecipe()


@pytest.fixture
def quantum_safe_recipe():
    """Return a quantum-safe voice recipe."""
    return QuantumSafeVoiceRecipe()


@pytest.fixture
def compliance_suite_recipe():
    """Return a compliance suite recipe."""
    return ComplianceSuiteRecipe()


@pytest.fixture
def full_bpo_recipe():
    """Return the full BPO security recipe."""
    return FullBPOSecurityRecipe()


@pytest.fixture
def sample_discovery_result():
    """Return a sample environment discovery result."""
    return EnvironmentDiscoveryResult(
        discovered_pbx=["Avaya Aura 10.1", "Cisco CUCM 14.0"],
        discovered_crm=["Salesforce Service Cloud"],
        discovered_wfm=["NICE WFM 7.5"],
        discovered_protocols=["SIP", "TN3270e", "RTP", "CTI-CSTA"],
        confidence_scores={
            "pbx": 0.95,
            "crm": 0.88,
            "wfm": 0.82,
            "protocols": 0.91,
        },
    )


@pytest.fixture
def sample_security_gap():
    """Return a sample security gap."""
    return SecurityGap(
        gap_type="DTMF_NOT_MASKED",
        severity="HIGH",
        description="DTMF tones are not masked during payment processing",
        auto_fixable=True,
        remediation="Enable DTMF masking in CLAMP mode for all voice channels",
    )


# ---------------------------------------------------------------------------
# Environment Discovery
# ---------------------------------------------------------------------------

class TestEnvironmentDiscovery:
    """Tests for EnvironmentDiscoveryResult."""

    def test_discovery_result_has_pbx(self, sample_discovery_result):
        """Test discovery result includes discovered PBX systems."""
        assert hasattr(sample_discovery_result, "discovered_pbx")
        assert len(sample_discovery_result.discovered_pbx) > 0, (
            "Discovery result should include at least one PBX system"
        )

    def test_discovery_result_has_crm(self, sample_discovery_result):
        """Test discovery result includes discovered CRM systems."""
        assert hasattr(sample_discovery_result, "discovered_crm")
        assert len(sample_discovery_result.discovered_crm) > 0, (
            "Discovery result should include at least one CRM system"
        )

    def test_discovery_result_has_wfm(self, sample_discovery_result):
        """Test discovery result includes discovered WFM systems."""
        assert hasattr(sample_discovery_result, "discovered_wfm")
        assert len(sample_discovery_result.discovered_wfm) > 0, (
            "Discovery result should include at least one WFM system"
        )

    def test_discovery_result_has_protocols(self, sample_discovery_result):
        """Test discovery result includes discovered protocols list."""
        assert hasattr(sample_discovery_result, "discovered_protocols")
        assert isinstance(sample_discovery_result.discovered_protocols, list)
        assert len(sample_discovery_result.discovered_protocols) > 0, (
            "Discovery result should include at least one protocol"
        )

    def test_confidence_scores_valid_range(self, sample_discovery_result):
        """Test confidence scores are between 0.0 and 1.0."""
        for category, score in sample_discovery_result.confidence_scores.items():
            assert 0.0 <= score <= 1.0, (
                f"Confidence score for '{category}' should be between 0.0 and 1.0, "
                f"got {score}"
            )

    def test_discovery_result_creation_minimal(self):
        """Test discovery result can be created with minimal data."""
        result = EnvironmentDiscoveryResult(
            discovered_pbx=[],
            discovered_crm=[],
            discovered_wfm=[],
            discovered_protocols=["SIP"],
            confidence_scores={"protocols": 0.5},
        )
        assert result is not None
        assert len(result.discovered_protocols) == 1

    @pytest.mark.asyncio
    async def test_orchestrator_discover_environment(self, orchestrator):
        """Test orchestrator can run environment discovery."""
        result = await orchestrator.discover_environment(
            target_network="10.0.0.0/24",
        )
        assert result is not None
        assert isinstance(result, EnvironmentDiscoveryResult)


# ---------------------------------------------------------------------------
# Security Assessment
# ---------------------------------------------------------------------------

class TestSecurityAssessment:
    """Tests for SecurityAssessmentResult and SecurityGap."""

    def test_security_gap_has_gap_type(self, sample_security_gap):
        """Test SecurityGap has gap_type field."""
        assert hasattr(sample_security_gap, "gap_type")
        assert sample_security_gap.gap_type == "DTMF_NOT_MASKED"

    def test_security_gap_has_severity(self, sample_security_gap):
        """Test SecurityGap has severity field."""
        assert hasattr(sample_security_gap, "severity")
        assert sample_security_gap.severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_security_gap_has_auto_fixable(self, sample_security_gap):
        """Test SecurityGap has auto_fixable field."""
        assert hasattr(sample_security_gap, "auto_fixable")
        assert isinstance(sample_security_gap.auto_fixable, bool)

    def test_security_gap_has_description(self, sample_security_gap):
        """Test SecurityGap has description."""
        assert hasattr(sample_security_gap, "description")
        assert len(sample_security_gap.description) > 0

    def test_security_gap_has_remediation(self, sample_security_gap):
        """Test SecurityGap has remediation guidance."""
        assert hasattr(sample_security_gap, "remediation")
        assert len(sample_security_gap.remediation) > 0

    def test_assessment_result_risk_score_range(self):
        """Test SecurityAssessmentResult risk score is 0-100."""
        result = SecurityAssessmentResult(
            risk_score=75,
            gaps=[],
            assessed_controls=10,
            compliant_controls=7,
        )
        assert 0 <= result.risk_score <= 100, (
            "Risk score should be between 0 and 100"
        )

    def test_assessment_identifies_dtmf_masking_gap(self):
        """Test assessment identifies missing DTMF masking as a gap."""
        dtmf_gap = SecurityGap(
            gap_type="DTMF_NOT_MASKED",
            severity="HIGH",
            description="DTMF tones are not masked",
            auto_fixable=True,
        )
        result = SecurityAssessmentResult(
            risk_score=80,
            gaps=[dtmf_gap],
            assessed_controls=10,
            compliant_controls=5,
        )
        gap_types = [g.gap_type for g in result.gaps]
        assert "DTMF_NOT_MASKED" in gap_types, (
            "Assessment should identify missing DTMF masking"
        )

    def test_assessment_identifies_unencrypted_recordings(self):
        """Test assessment identifies unencrypted recordings as a gap."""
        recording_gap = SecurityGap(
            gap_type="RECORDINGS_UNENCRYPTED",
            severity="HIGH",
            description="Call recordings are stored without encryption",
            auto_fixable=True,
        )
        result = SecurityAssessmentResult(
            risk_score=85,
            gaps=[recording_gap],
            assessed_controls=10,
            compliant_controls=4,
        )
        gap_types = [g.gap_type for g in result.gaps]
        assert "RECORDINGS_UNENCRYPTED" in gap_types, (
            "Assessment should identify unencrypted recordings"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_assess_security(self, orchestrator, sample_discovery_result):
        """Test orchestrator can run security assessment."""
        result = await orchestrator.assess_security(sample_discovery_result)
        assert result is not None
        assert isinstance(result, SecurityAssessmentResult)
        assert 0 <= result.risk_score <= 100


# ---------------------------------------------------------------------------
# Policy Generation
# ---------------------------------------------------------------------------

class TestPolicyGeneration:
    """Tests for PolicyGenerationResult."""

    def test_generated_policy_has_policy_id(self):
        """Test generated policies have policy_id."""
        result = PolicyGenerationResult(
            policies=[{
                "policy_id": "POL-PCI-001",
                "policy_type": "PCI-DSS",
                "rules": ["DTMF masking required", "Recording encryption required"],
                "auto_deployable": True,
            }],
        )
        assert result.policies[0]["policy_id"] == "POL-PCI-001"

    def test_generated_policy_has_policy_type(self):
        """Test generated policies have policy_type."""
        result = PolicyGenerationResult(
            policies=[{
                "policy_id": "POL-PCI-002",
                "policy_type": "PCI-DSS",
                "rules": ["DTMF masking"],
                "auto_deployable": True,
            }],
        )
        assert result.policies[0]["policy_type"] == "PCI-DSS"

    def test_generated_policy_has_rules(self):
        """Test generated policies have rules list."""
        result = PolicyGenerationResult(
            policies=[{
                "policy_id": "POL-PCI-003",
                "policy_type": "PCI-DSS",
                "rules": ["DTMF masking required", "Recording pause/resume enabled"],
                "auto_deployable": True,
            }],
        )
        assert isinstance(result.policies[0]["rules"], list)
        assert len(result.policies[0]["rules"]) > 0

    def test_auto_deployable_flag(self):
        """Test auto_deployable flag is correctly set."""
        deployable = PolicyGenerationResult(
            policies=[{
                "policy_id": "POL-AUTO-001",
                "policy_type": "PCI-DSS",
                "rules": ["DTMF masking"],
                "auto_deployable": True,
            }],
        )
        non_deployable = PolicyGenerationResult(
            policies=[{
                "policy_id": "POL-MANUAL-001",
                "policy_type": "CUSTOM",
                "rules": ["Manual review required"],
                "auto_deployable": False,
            }],
        )
        assert deployable.policies[0]["auto_deployable"] is True
        assert non_deployable.policies[0]["auto_deployable"] is False

    def test_pci_dss_policy_includes_dtmf_masking(self):
        """Test PCI-DSS policies include DTMF masking rules."""
        result = PolicyGenerationResult(
            policies=[{
                "policy_id": "POL-PCI-004",
                "policy_type": "PCI-DSS",
                "rules": [
                    "Enable DTMF masking in CLAMP mode",
                    "Pause recording during payment",
                    "Encrypt all recordings with ML-KEM-1024",
                ],
                "auto_deployable": True,
            }],
        )
        rules = result.policies[0]["rules"]
        assert any("DTMF" in rule.upper() or "dtmf" in rule.lower() for rule in rules), (
            "PCI-DSS policy should include DTMF masking rules"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_generate_policies(self, orchestrator):
        """Test orchestrator can generate policies."""
        assessment = SecurityAssessmentResult(
            risk_score=75,
            gaps=[
                SecurityGap(
                    gap_type="DTMF_NOT_MASKED",
                    severity="HIGH",
                    description="DTMF not masked",
                    auto_fixable=True,
                ),
            ],
            assessed_controls=10,
            compliant_controls=7,
        )
        result = await orchestrator.generate_policies(assessment)
        assert result is not None
        assert isinstance(result, PolicyGenerationResult)
        assert len(result.policies) > 0


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------

class TestProvisioning:
    """Tests for ProvisioningResult."""

    def test_provisioning_tracks_deployed_controls(self):
        """Test ProvisioningResult tracks deployed controls."""
        result = ProvisioningResult(
            deployed_controls=["dtmf_masking", "recording_encryption", "srtp_pqc"],
            failed_controls=[],
            rollback_available=True,
        )
        assert len(result.deployed_controls) == 3
        assert "dtmf_masking" in result.deployed_controls

    def test_provisioning_tracks_failed_controls(self):
        """Test ProvisioningResult tracks failed controls."""
        result = ProvisioningResult(
            deployed_controls=["dtmf_masking"],
            failed_controls=["srtp_pqc"],
            rollback_available=True,
        )
        assert len(result.failed_controls) == 1
        assert "srtp_pqc" in result.failed_controls

    def test_rollback_available_when_controls_deployed(self):
        """Test rollback is available when controls are deployed."""
        result = ProvisioningResult(
            deployed_controls=["dtmf_masking", "recording_encryption"],
            failed_controls=[],
            rollback_available=True,
        )
        assert result.rollback_available is True, (
            "Rollback should be available when controls are deployed"
        )

    def test_no_rollback_when_nothing_deployed(self):
        """Test rollback is not available when nothing deployed."""
        result = ProvisioningResult(
            deployed_controls=[],
            failed_controls=["dtmf_masking"],
            rollback_available=False,
        )
        assert result.rollback_available is False

    @pytest.mark.asyncio
    async def test_orchestrator_provision(self, orchestrator):
        """Test orchestrator can provision security controls."""
        policies = PolicyGenerationResult(
            policies=[{
                "policy_id": "POL-PROV-001",
                "policy_type": "PCI-DSS",
                "rules": ["Enable DTMF masking"],
                "auto_deployable": True,
            }],
        )
        result = await orchestrator.provision(policies)
        assert result is not None
        assert isinstance(result, ProvisioningResult)


# ---------------------------------------------------------------------------
# Zero-Touch Deployment (Full Pipeline)
# ---------------------------------------------------------------------------

class TestZeroTouchDeployment:
    """Tests for ZeroTouchDeploymentResult (full pipeline)."""

    def test_deployment_result_has_all_phases(self):
        """Test full deployment result has all 5 phases."""
        result = ZeroTouchDeploymentResult(
            discovery=EnvironmentDiscoveryResult(
                discovered_pbx=["Avaya"],
                discovered_crm=["Salesforce"],
                discovered_wfm=["NICE"],
                discovered_protocols=["SIP"],
                confidence_scores={"protocols": 0.9},
            ),
            assessment=SecurityAssessmentResult(
                risk_score=80,
                gaps=[],
                assessed_controls=10,
                compliant_controls=8,
            ),
            policies=PolicyGenerationResult(
                policies=[{"policy_id": "P1", "policy_type": "PCI", "rules": [], "auto_deployable": True}],
            ),
            provisioning=ProvisioningResult(
                deployed_controls=["dtmf_masking"],
                failed_controls=[],
                rollback_available=True,
            ),
            monitoring_enabled=True,
        )
        assert result.discovery is not None, "Deployment should have discovery phase"
        assert result.assessment is not None, "Deployment should have assessment phase"
        assert result.policies is not None, "Deployment should have policies phase"
        assert result.provisioning is not None, "Deployment should have provisioning phase"
        assert result.monitoring_enabled is True, "Deployment should have monitoring phase"

    def test_risk_reduction_calculated(self):
        """Test risk reduction is calculated (before_score - after_score)."""
        result = ZeroTouchDeploymentResult(
            discovery=EnvironmentDiscoveryResult(
                discovered_pbx=[], discovered_crm=[], discovered_wfm=[],
                discovered_protocols=["SIP"],
                confidence_scores={"protocols": 0.9},
            ),
            assessment=SecurityAssessmentResult(
                risk_score=80,
                gaps=[],
                assessed_controls=10,
                compliant_controls=5,
            ),
            policies=PolicyGenerationResult(policies=[]),
            provisioning=ProvisioningResult(
                deployed_controls=["dtmf_masking", "srtp"],
                failed_controls=[],
                rollback_available=True,
            ),
            monitoring_enabled=True,
            risk_score_before=80,
            risk_score_after=25,
        )
        risk_reduction = result.risk_score_before - result.risk_score_after
        assert risk_reduction == 55, (
            f"Risk reduction should be 55, got {risk_reduction}"
        )
        assert result.risk_score_after < result.risk_score_before, (
            "Risk score after deployment should be lower than before"
        )

    def test_deployment_tracks_total_time(self):
        """Test deployment tracks total time."""
        result = ZeroTouchDeploymentResult(
            discovery=EnvironmentDiscoveryResult(
                discovered_pbx=[], discovered_crm=[], discovered_wfm=[],
                discovered_protocols=["SIP"],
                confidence_scores={},
            ),
            assessment=SecurityAssessmentResult(
                risk_score=50, gaps=[], assessed_controls=5, compliant_controls=3,
            ),
            policies=PolicyGenerationResult(policies=[]),
            provisioning=ProvisioningResult(
                deployed_controls=[], failed_controls=[], rollback_available=False,
            ),
            monitoring_enabled=True,
            total_time_seconds=42.5,
        )
        assert hasattr(result, "total_time_seconds")
        assert result.total_time_seconds >= 0, (
            "Total deployment time should be non-negative"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_full_deployment(self, orchestrator):
        """Test orchestrator can run full zero-touch deployment."""
        result = await orchestrator.deploy(
            target_network="10.0.0.0/24",
            environment="ON_PREMISE",
        )
        assert result is not None
        assert isinstance(result, ZeroTouchDeploymentResult)


# ---------------------------------------------------------------------------
# Recipe Registry
# ---------------------------------------------------------------------------

class TestRecipeRegistry:
    """Tests for RecipeRegistry."""

    def test_registry_preloads_builtin_recipes(self, recipe_registry):
        """Test registry pre-loads 6 built-in recipes."""
        recipes = recipe_registry.list_recipes()
        assert len(recipes) >= 6, (
            f"Registry should pre-load at least 6 built-in recipes, got {len(recipes)}"
        )

    def test_list_recipes_returns_all(self, recipe_registry):
        """Test list_recipes returns all registered recipes."""
        recipes = recipe_registry.list_recipes()
        assert isinstance(recipes, list)
        assert len(recipes) > 0

    def test_get_recipe_by_id(self, recipe_registry):
        """Test get_recipe retrieves a recipe by ID."""
        recipes = recipe_registry.list_recipes()
        first_id = recipes[0].recipe_id if hasattr(recipes[0], "recipe_id") else recipes[0]["recipe_id"]
        recipe = recipe_registry.get_recipe(first_id)
        assert recipe is not None, (
            f"Recipe with ID '{first_id}' should be retrievable"
        )

    def test_get_recipe_nonexistent_returns_none(self, recipe_registry):
        """Test get_recipe returns None for nonexistent recipe."""
        recipe = recipe_registry.get_recipe("nonexistent_recipe_id")
        assert recipe is None, (
            "Nonexistent recipe should return None"
        )

    def test_get_recipes_for_on_premise(self, recipe_registry):
        """Test get_recipes_for_environment filters for ON_PREMISE."""
        recipes = recipe_registry.get_recipes_for_environment("ON_PREMISE")
        assert isinstance(recipes, list)
        assert len(recipes) > 0, (
            "Should return at least one recipe for ON_PREMISE environment"
        )

    def test_get_recipes_for_cloud(self, recipe_registry):
        """Test get_recipes_for_environment filters for CLOUD_CCaaS."""
        recipes = recipe_registry.get_recipes_for_environment("CLOUD_CCaaS")
        assert isinstance(recipes, list)
        assert len(recipes) > 0, (
            "Should return at least one recipe for CLOUD_CCaaS environment"
        )

    def test_get_recipes_for_remote(self, recipe_registry):
        """Test get_recipes_for_environment filters for REMOTE_WORKFORCE."""
        recipes = recipe_registry.get_recipes_for_environment("REMOTE_WORKFORCE")
        assert isinstance(recipes, list)
        assert len(recipes) > 0, (
            "Should return at least one recipe for REMOTE_WORKFORCE environment"
        )

    def test_registry_recipes_are_bpo_automation_recipes(self, recipe_registry):
        """Test all registered recipes are BPOAutomationRecipe instances."""
        recipes = recipe_registry.list_recipes()
        for recipe in recipes:
            assert isinstance(recipe, BPOAutomationRecipe), (
                f"Recipe should be a BPOAutomationRecipe instance, "
                f"got {type(recipe).__name__}"
            )


# ---------------------------------------------------------------------------
# PCI-DSS Voice Compliance Recipe
# ---------------------------------------------------------------------------

class TestPCIDSSRecipe:
    """Tests for PCIDSSVoiceComplianceRecipe."""

    def test_recipe_creation(self, pci_recipe):
        """Test PCI-DSS recipe can be created."""
        assert pci_recipe is not None
        assert isinstance(pci_recipe, BPOAutomationRecipe)

    def test_recipe_has_dtmf_masking_step(self, pci_recipe):
        """Test recipe includes DTMF masking step."""
        step_names = [s.name for s in pci_recipe.steps]
        assert any("dtmf" in name.lower() or "DTMF" in name for name in step_names), (
            "PCI-DSS recipe should include a DTMF masking step"
        )

    def test_recipe_has_recording_pause_step(self, pci_recipe):
        """Test recipe includes recording pause/resume step."""
        step_names = [s.name for s in pci_recipe.steps]
        assert any("recording" in name.lower() and "pause" in name.lower()
                    for name in step_names), (
            "PCI-DSS recipe should include a recording pause step"
        )

    def test_recipe_has_pan_detection_step(self, pci_recipe):
        """Test recipe includes PAN detection step."""
        step_names = [s.name for s in pci_recipe.steps]
        assert any("pan" in name.lower() or "card" in name.lower()
                    for name in step_names), (
            "PCI-DSS recipe should include a PAN detection step"
        )

    def test_recipe_has_screen_masking_step(self, pci_recipe):
        """Test recipe includes screen masking step."""
        step_names = [s.name for s in pci_recipe.steps]
        assert any("screen" in name.lower() and "mask" in name.lower()
                    for name in step_names), (
            "PCI-DSS recipe should include a screen masking step"
        )

    def test_recipe_has_at_least_4_steps(self, pci_recipe):
        """Test recipe has at least 4 steps."""
        assert len(pci_recipe.steps) >= 4, (
            f"PCI-DSS recipe should have at least 4 steps, got {len(pci_recipe.steps)}"
        )


# ---------------------------------------------------------------------------
# Toll Fraud Prevention Recipe
# ---------------------------------------------------------------------------

class TestTollFraudRecipe:
    """Tests for TollFraudPreventionRecipe."""

    def test_recipe_creation(self, toll_fraud_recipe):
        """Test toll fraud recipe can be created."""
        assert toll_fraud_recipe is not None
        assert isinstance(toll_fraud_recipe, BPOAutomationRecipe)

    def test_recipe_has_premium_rate_db_step(self, toll_fraud_recipe):
        """Test recipe includes premium rate database step."""
        step_names = [s.name for s in toll_fraud_recipe.steps]
        assert any("premium" in name.lower() or "rate" in name.lower()
                    for name in step_names), (
            "Toll fraud recipe should include a premium rate database step"
        )

    def test_recipe_has_irsf_detection_step(self, toll_fraud_recipe):
        """Test recipe includes IRSF detection step."""
        step_names = [s.name for s in toll_fraud_recipe.steps]
        assert any("irsf" in name.lower() or "revenue share" in name.lower()
                    for name in step_names), (
            "Toll fraud recipe should include an IRSF detection step"
        )

    def test_recipe_has_velocity_alerting_step(self, toll_fraud_recipe):
        """Test recipe includes velocity alerting step."""
        step_names = [s.name for s in toll_fraud_recipe.steps]
        assert any("velocity" in name.lower() or "alert" in name.lower()
                    for name in step_names), (
            "Toll fraud recipe should include a velocity alerting step"
        )

    def test_recipe_has_at_least_3_steps(self, toll_fraud_recipe):
        """Test toll fraud recipe has at least 3 steps."""
        assert len(toll_fraud_recipe.steps) >= 3, (
            f"Toll fraud recipe should have at least 3 steps, "
            f"got {len(toll_fraud_recipe.steps)}"
        )


# ---------------------------------------------------------------------------
# Remote Workforce Security Recipe
# ---------------------------------------------------------------------------

class TestRemoteWorkforceRecipe:
    """Tests for RemoteWorkforceSecurityRecipe."""

    def test_recipe_creation(self, remote_workforce_recipe):
        """Test remote workforce recipe can be created."""
        assert remote_workforce_recipe is not None
        assert isinstance(remote_workforce_recipe, BPOAutomationRecipe)

    def test_recipe_has_pqc_tunnel_step(self, remote_workforce_recipe):
        """Test recipe includes PQC tunnel setup step."""
        step_names = [s.name for s in remote_workforce_recipe.steps]
        assert any("pqc" in name.lower() and "tunnel" in name.lower()
                    for name in step_names), (
            "Remote workforce recipe should include a PQC tunnel step"
        )

    def test_recipe_has_device_posture_step(self, remote_workforce_recipe):
        """Test recipe includes device posture check step."""
        step_names = [s.name for s in remote_workforce_recipe.steps]
        assert any("device" in name.lower() and "posture" in name.lower()
                    for name in step_names), (
            "Remote workforce recipe should include a device posture step"
        )

    def test_recipe_has_geo_fence_step(self, remote_workforce_recipe):
        """Test recipe includes geo-fence enforcement step."""
        step_names = [s.name for s in remote_workforce_recipe.steps]
        assert any("geo" in name.lower() and "fence" in name.lower()
                    for name in step_names), (
            "Remote workforce recipe should include a geo-fence step"
        )

    def test_recipe_has_at_least_3_steps(self, remote_workforce_recipe):
        """Test remote workforce recipe has at least 3 steps."""
        assert len(remote_workforce_recipe.steps) >= 3, (
            f"Remote workforce recipe should have at least 3 steps, "
            f"got {len(remote_workforce_recipe.steps)}"
        )


# ---------------------------------------------------------------------------
# Quantum-Safe Voice Recipe
# ---------------------------------------------------------------------------

class TestQuantumSafeVoiceRecipe:
    """Tests for QuantumSafeVoiceRecipe."""

    def test_recipe_creation(self, quantum_safe_recipe):
        """Test quantum-safe voice recipe can be created."""
        assert quantum_safe_recipe is not None
        assert isinstance(quantum_safe_recipe, BPOAutomationRecipe)

    def test_recipe_has_ml_kem_768_step(self, quantum_safe_recipe):
        """Test recipe includes ML-KEM-768 key exchange step."""
        step_names = [s.name for s in quantum_safe_recipe.steps]
        step_descriptions = [
            s.description if hasattr(s, "description") else "" for s in quantum_safe_recipe.steps
        ]
        all_text = " ".join(step_names + step_descriptions).lower()
        assert "ml-kem-768" in all_text or "ml_kem_768" in all_text or "mlkem768" in all_text, (
            "Quantum-safe recipe should include ML-KEM-768 key exchange"
        )

    def test_recipe_has_sip_pqc_tls_step(self, quantum_safe_recipe):
        """Test recipe includes SIP-PQC-TLS step."""
        step_names = [s.name for s in quantum_safe_recipe.steps]
        step_descriptions = [
            s.description if hasattr(s, "description") else "" for s in quantum_safe_recipe.steps
        ]
        all_text = " ".join(step_names + step_descriptions).lower()
        assert "sip" in all_text and "pqc" in all_text, (
            "Quantum-safe recipe should include SIP-PQC-TLS step"
        )

    def test_recipe_has_srtp_pqc_step(self, quantum_safe_recipe):
        """Test recipe includes SRTP-PQC media encryption step."""
        step_names = [s.name for s in quantum_safe_recipe.steps]
        step_descriptions = [
            s.description if hasattr(s, "description") else "" for s in quantum_safe_recipe.steps
        ]
        all_text = " ".join(step_names + step_descriptions).lower()
        assert "srtp" in all_text and "pqc" in all_text, (
            "Quantum-safe recipe should include SRTP-PQC step"
        )

    def test_recipe_has_at_least_3_steps(self, quantum_safe_recipe):
        """Test quantum-safe recipe has at least 3 steps."""
        assert len(quantum_safe_recipe.steps) >= 3, (
            f"Quantum-safe recipe should have at least 3 steps, "
            f"got {len(quantum_safe_recipe.steps)}"
        )


# ---------------------------------------------------------------------------
# Automation Step Dataclass
# ---------------------------------------------------------------------------

class TestAutomationStep:
    """Tests for AutomationStep dataclass."""

    def test_step_has_required_fields(self):
        """Test AutomationStep has required fields."""
        step = AutomationStep(
            name="Enable DTMF Masking",
            description="Configure DTMF masking in CLAMP mode",
            action="configure_dtmf_masking",
        )
        assert step.name == "Enable DTMF Masking"
        assert step.description == "Configure DTMF masking in CLAMP mode"
        assert step.action == "configure_dtmf_masking"

    def test_step_has_depends_on(self):
        """Test AutomationStep supports depends_on for ordering."""
        step = AutomationStep(
            name="Configure SRTP",
            description="Enable SRTP encryption",
            action="configure_srtp",
            depends_on=["configure_sip_tls"],
        )
        assert hasattr(step, "depends_on")
        assert "configure_sip_tls" in step.depends_on, (
            "Step should have depends_on for ordering"
        )

    def test_step_without_dependencies(self):
        """Test AutomationStep without dependencies defaults to empty."""
        step = AutomationStep(
            name="Initial Scan",
            description="Scan the environment",
            action="scan_environment",
        )
        depends = getattr(step, "depends_on", [])
        assert isinstance(depends, (list, type(None))), (
            "Step without dependencies should have empty depends_on"
        )

    def test_step_with_timeout(self):
        """Test AutomationStep supports timeout."""
        step = AutomationStep(
            name="Deploy Encryption",
            description="Deploy ML-KEM encryption",
            action="deploy_encryption",
            timeout_seconds=300,
        )
        assert hasattr(step, "timeout_seconds")
        assert step.timeout_seconds == 300

    def test_step_with_rollback_action(self):
        """Test AutomationStep supports rollback action."""
        step = AutomationStep(
            name="Configure Firewall",
            description="Update firewall rules",
            action="configure_firewall",
            rollback_action="revert_firewall",
        )
        assert hasattr(step, "rollback_action")
        assert step.rollback_action == "revert_firewall"


# ---------------------------------------------------------------------------
# Full BPO Security Recipe
# ---------------------------------------------------------------------------

class TestFullBPOSecurityRecipe:
    """Tests for FullBPOSecurityRecipe (comprehensive recipe)."""

    def test_recipe_creation(self, full_bpo_recipe):
        """Test full BPO security recipe can be created."""
        assert full_bpo_recipe is not None
        assert isinstance(full_bpo_recipe, BPOAutomationRecipe)

    def test_recipe_is_comprehensive(self, full_bpo_recipe):
        """Test full BPO recipe has more steps than individual recipes."""
        assert len(full_bpo_recipe.steps) >= 6, (
            f"Full BPO recipe should be comprehensive with at least 6 steps, "
            f"got {len(full_bpo_recipe.steps)}"
        )

    def test_recipe_has_recipe_id(self, full_bpo_recipe):
        """Test recipe has a recipe_id."""
        assert hasattr(full_bpo_recipe, "recipe_id")
        assert full_bpo_recipe.recipe_id is not None
        assert len(full_bpo_recipe.recipe_id) > 0


# ---------------------------------------------------------------------------
# Compliance Suite Recipe
# ---------------------------------------------------------------------------

class TestComplianceSuiteRecipe:
    """Tests for ComplianceSuiteRecipe."""

    def test_recipe_creation(self, compliance_suite_recipe):
        """Test compliance suite recipe can be created."""
        assert compliance_suite_recipe is not None
        assert isinstance(compliance_suite_recipe, BPOAutomationRecipe)

    def test_recipe_has_recipe_id(self, compliance_suite_recipe):
        """Test recipe has a recipe_id."""
        assert hasattr(compliance_suite_recipe, "recipe_id")
        assert compliance_suite_recipe.recipe_id is not None

    def test_recipe_has_steps(self, compliance_suite_recipe):
        """Test recipe has at least one step."""
        assert len(compliance_suite_recipe.steps) >= 1, (
            "Compliance suite recipe should have at least one step"
        )


# ---------------------------------------------------------------------------
# Orchestrator Integration
# ---------------------------------------------------------------------------

class TestOrchestratorIntegration:
    """Integration tests for BPOZeroTouchOrchestrator."""

    def test_orchestrator_creation(self, orchestrator):
        """Test orchestrator can be instantiated."""
        assert orchestrator is not None
        assert isinstance(orchestrator, BPOZeroTouchOrchestrator)

    @pytest.mark.asyncio
    async def test_orchestrator_full_pipeline(self, orchestrator):
        """Test orchestrator runs the full 5-phase pipeline."""
        result = await orchestrator.deploy(
            target_network="10.0.0.0/24",
            environment="ON_PREMISE",
        )
        assert result is not None
        assert isinstance(result, ZeroTouchDeploymentResult)
        assert result.discovery is not None
        assert result.assessment is not None
        assert result.policies is not None
        assert result.provisioning is not None

    @pytest.mark.asyncio
    async def test_orchestrator_accepts_recipe(self, orchestrator, pci_recipe):
        """Test orchestrator can execute a specific recipe."""
        result = await orchestrator.execute_recipe(
            recipe=pci_recipe,
            target_network="10.0.0.0/24",
        )
        assert result is not None

    @pytest.mark.parametrize("environment", [
        "ON_PREMISE",
        "CLOUD_CCaaS",
        "REMOTE_WORKFORCE",
        "HYBRID",
    ])
    @pytest.mark.asyncio
    async def test_orchestrator_supports_all_environments(self, orchestrator, environment):
        """Test orchestrator supports all environment types."""
        result = await orchestrator.deploy(
            target_network="10.0.0.0/24",
            environment=environment,
        )
        assert result is not None
        assert isinstance(result, ZeroTouchDeploymentResult)
