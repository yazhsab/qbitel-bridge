"""
Tests for BPO Compliance Context Module

Tests cover:
- BPOComplianceFramework enum
- Control initialization for each framework
- update_control_status()
- log_event() and log_call_event()
- log_dtmf_masking_event()
- log_recording_pause_resume()
- Audit trail querying (by time, agent, call_id)
- verify_audit_chain() integrity
- get_compliance_summary()
- Multi-tenant isolation
"""

import pytest
from datetime import datetime, timedelta
from uuid import UUID

from ai_engine.domains.bpo.core.compliance_context import (
    BPOComplianceFramework,
    ComplianceStatus,
    BPOControlCategory,
    ComplianceControl,
    BPOAuditEvent,
    BPOComplianceContext,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_context():
    """Return a default BPOComplianceContext."""
    return BPOComplianceContext()


@pytest.fixture
def pci_context():
    """Return a PCI-DSS focused context."""
    return BPOComplianceContext(
        frameworks={BPOComplianceFramework.PCI_DSS_4_0},
        organization_id="ORG-001",
    )


@pytest.fixture
def hipaa_context():
    """Return a HIPAA focused context."""
    return BPOComplianceContext(
        frameworks={BPOComplianceFramework.HIPAA},
        organization_id="ORG-002",
    )


@pytest.fixture
def tcpa_context():
    """Return a TCPA focused context."""
    return BPOComplianceContext(
        frameworks={BPOComplianceFramework.TCPA},
        organization_id="ORG-003",
    )


@pytest.fixture
def multi_framework_context():
    """Return a multi-framework context."""
    return BPOComplianceContext(
        frameworks={
            BPOComplianceFramework.PCI_DSS_4_0,
            BPOComplianceFramework.HIPAA,
            BPOComplianceFramework.TCPA,
            BPOComplianceFramework.SOC_2,
            BPOComplianceFramework.GDPR,
            BPOComplianceFramework.NIST_PQC,
        },
        organization_id="ORG-MULTI",
    )


@pytest.fixture
def tenant_context():
    """Return a tenant-specific context."""
    return BPOComplianceContext(
        frameworks={BPOComplianceFramework.PCI_DSS_4_0},
        organization_id="ORG-001",
        tenant_id="TENANT-A",
    )


# ---------------------------------------------------------------------------
# BPOComplianceFramework Enum
# ---------------------------------------------------------------------------

class TestBPOComplianceFramework:
    """Tests for BPOComplianceFramework enumeration."""

    def test_pci_dss_properties(self):
        """Test PCI-DSS framework properties."""
        fw = BPOComplianceFramework.PCI_DSS_4_0
        assert fw.framework == "PCI-DSS"
        assert fw.version == "4.0"
        assert "Payment Card" in fw.description

    def test_tcpa_properties(self):
        """Test TCPA framework properties."""
        fw = BPOComplianceFramework.TCPA
        assert fw.framework == "TCPA"
        assert "Telephone" in fw.description

    def test_hipaa_properties(self):
        """Test HIPAA framework properties."""
        fw = BPOComplianceFramework.HIPAA
        assert fw.framework == "HIPAA"
        assert "Health" in fw.description

    def test_soc2_properties(self):
        """Test SOC 2 framework properties."""
        fw = BPOComplianceFramework.SOC_2
        assert fw.framework == "SOC 2"
        assert "Service Organization" in fw.description

    def test_gdpr_properties(self):
        """Test GDPR framework properties."""
        fw = BPOComplianceFramework.GDPR
        assert fw.framework == "GDPR"
        assert "Data Protection" in fw.description

    def test_nist_pqc_properties(self):
        """Test NIST PQC framework properties."""
        fw = BPOComplianceFramework.NIST_PQC
        assert fw.framework == "NIST"
        assert "Quantum" in fw.description

    def test_full_name(self):
        """Test full_name property."""
        fw = BPOComplianceFramework.PCI_DSS_4_0
        assert fw.full_name == "PCI-DSS 4.0"

    @pytest.mark.parametrize("framework", list(BPOComplianceFramework))
    def test_all_frameworks_have_description(self, framework):
        """Test that all frameworks have non-empty descriptions."""
        assert len(framework.description) > 0
        assert len(framework.full_name) > 0


# ---------------------------------------------------------------------------
# Control Initialization
# ---------------------------------------------------------------------------

class TestControlInitialization:
    """Tests for compliance control initialization."""

    def test_pci_controls_initialized(self, pci_context):
        """Test PCI-DSS controls are initialized."""
        control = pci_context.get_control("PCI-CC-3.4.1")
        assert control is not None
        assert control.framework == BPOComplianceFramework.PCI_DSS_4_0
        assert "DTMF" in control.title

    def test_pci_dtmf_masking_control(self, pci_context):
        """Test PCI DTMF masking control exists."""
        control = pci_context.get_control("PCI-CC-3.4.1")
        assert control is not None
        assert control.category == BPOControlCategory.VOICE_SECURITY

    def test_pci_recording_encryption_control(self, pci_context):
        """Test PCI call recording encryption control exists."""
        control = pci_context.get_control("PCI-CC-3.5.1")
        assert control is not None
        assert control.category == BPOControlCategory.CALL_RECORDING

    def test_pci_voice_encryption_control(self, pci_context):
        """Test PCI voice channel encryption control exists."""
        control = pci_context.get_control("PCI-CC-4.2.1")
        assert control is not None
        assert control.category == BPOControlCategory.CRYPTOGRAPHY

    def test_pci_agent_auth_control(self, pci_context):
        """Test PCI agent authentication control exists."""
        control = pci_context.get_control("PCI-CC-8.3.1")
        assert control is not None
        assert control.category == BPOControlCategory.ACCESS_CONTROL

    def test_hipaa_controls_initialized(self, hipaa_context):
        """Test HIPAA controls are initialized."""
        control = hipaa_context.get_control("HIPAA-CC-1.1")
        assert control is not None
        assert control.framework == BPOComplianceFramework.HIPAA
        assert "PHI" in control.title

    def test_hipaa_remote_access_control(self, hipaa_context):
        """Test HIPAA remote access control exists."""
        control = hipaa_context.get_control("HIPAA-CC-3.1")
        assert control is not None
        assert control.category == BPOControlCategory.REMOTE_ACCESS

    def test_tcpa_controls_initialized(self, tcpa_context):
        """Test TCPA controls are initialized."""
        control = tcpa_context.get_control("TCPA-1.1")
        assert control is not None
        assert control.framework == BPOComplianceFramework.TCPA
        assert "consent" in control.title.lower()

    def test_tcpa_dnc_control(self, tcpa_context):
        """Test TCPA Do-Not-Call control exists."""
        control = tcpa_context.get_control("TCPA-1.2")
        assert control is not None
        assert "DNC" in control.title or "Do-Not-Call" in control.title

    def test_multi_framework_all_controls(self, multi_framework_context):
        """Test multi-framework context has controls from all frameworks."""
        # Check at least one control from each framework
        assert multi_framework_context.get_control("PCI-CC-3.4.1") is not None
        assert multi_framework_context.get_control("HIPAA-CC-1.1") is not None
        assert multi_framework_context.get_control("TCPA-1.1") is not None
        assert multi_framework_context.get_control("SOC2-CC-6.1") is not None
        assert multi_framework_context.get_control("GDPR-CC-1.1") is not None
        assert multi_framework_context.get_control("PQC-BPO-1.1") is not None

    def test_nonexistent_control_returns_none(self, pci_context):
        """Test getting nonexistent control returns None."""
        assert pci_context.get_control("NONEXISTENT-1.1") is None

    def test_controls_default_to_under_review(self, pci_context):
        """Test newly initialized controls default to UNDER_REVIEW."""
        control = pci_context.get_control("PCI-CC-3.4.1")
        assert control.status == ComplianceStatus.UNDER_REVIEW


# ---------------------------------------------------------------------------
# update_control_status()
# ---------------------------------------------------------------------------

class TestUpdateControlStatus:
    """Tests for update_control_status()."""

    def test_update_to_compliant(self, pci_context):
        """Test updating control to COMPLIANT."""
        result = pci_context.update_control_status(
            control_id="PCI-CC-3.4.1",
            status=ComplianceStatus.COMPLIANT,
            assessed_by="auditor@example.com",
        )
        assert result is True
        control = pci_context.get_control("PCI-CC-3.4.1")
        assert control.status == ComplianceStatus.COMPLIANT
        assert control.assessed_by == "auditor@example.com"
        assert control.last_assessed is not None

    def test_update_to_non_compliant(self, pci_context):
        """Test updating control to NON_COMPLIANT."""
        result = pci_context.update_control_status(
            control_id="PCI-CC-3.4.1",
            status=ComplianceStatus.NON_COMPLIANT,
            assessed_by="auditor@example.com",
        )
        assert result is True
        control = pci_context.get_control("PCI-CC-3.4.1")
        assert control.status == ComplianceStatus.NON_COMPLIANT

    def test_update_with_evidence(self, pci_context):
        """Test updating control with evidence references."""
        evidence = ["scan_report_2024.pdf", "pentest_results.pdf"]
        result = pci_context.update_control_status(
            control_id="PCI-CC-3.4.1",
            status=ComplianceStatus.COMPLIANT,
            assessed_by="auditor@example.com",
            evidence=evidence,
        )
        assert result is True
        control = pci_context.get_control("PCI-CC-3.4.1")
        assert "scan_report_2024.pdf" in control.evidence_references
        assert "pentest_results.pdf" in control.evidence_references

    def test_update_nonexistent_control_returns_false(self, pci_context):
        """Test updating nonexistent control returns False."""
        result = pci_context.update_control_status(
            control_id="NONEXISTENT-1.1",
            status=ComplianceStatus.COMPLIANT,
            assessed_by="auditor@example.com",
        )
        assert result is False

    def test_update_generates_audit_event(self, pci_context):
        """Test that updating a control generates an audit event."""
        pci_context.update_control_status(
            control_id="PCI-CC-3.4.1",
            status=ComplianceStatus.COMPLIANT,
            assessed_by="auditor@example.com",
        )
        events = pci_context.get_audit_trail(event_type="CONTROL_ASSESSMENT")
        assert len(events) >= 1
        assert "PCI-CC-3.4.1" in events[-1].description


# ---------------------------------------------------------------------------
# log_event() and log_call_event()
# ---------------------------------------------------------------------------

class TestLogging:
    """Tests for log_event() and log_call_event()."""

    def test_log_event_returns_audit_event(self, default_context):
        """Test log_event returns a BPOAuditEvent."""
        event = default_context.log_event(
            event_type="TEST_EVENT",
            event_category="TEST",
            description="Test event",
            action="TEST_ACTION",
        )
        assert isinstance(event, BPOAuditEvent)
        assert event.event_type == "TEST_EVENT"
        assert event.event_category == "TEST"

    def test_log_event_has_uuid(self, default_context):
        """Test logged event has a UUID."""
        event = default_context.log_event(
            event_type="TEST",
            event_category="TEST",
            description="Test",
            action="TEST",
        )
        assert isinstance(event.event_id, UUID)

    def test_log_event_has_timestamp(self, default_context):
        """Test logged event has a timestamp."""
        before = datetime.utcnow()
        event = default_context.log_event(
            event_type="TEST",
            event_category="TEST",
            description="Test",
            action="TEST",
        )
        after = datetime.utcnow()
        assert before <= event.timestamp <= after

    def test_log_event_with_agent_id(self, default_context):
        """Test logging event with agent ID."""
        event = default_context.log_event(
            event_type="AGENT_LOGIN",
            event_category="AUTH",
            description="Agent logged in",
            action="LOGIN",
            agent_id="AGENT-001",
        )
        assert event.agent_id == "AGENT-001"

    def test_log_event_with_call_id(self, default_context):
        """Test logging event with call ID."""
        event = default_context.log_event(
            event_type="CALL_START",
            event_category="CALL",
            description="Call started",
            action="CALL_START",
            call_id="CALL-12345",
        )
        assert event.call_id == "CALL-12345"

    def test_log_call_event(self, default_context):
        """Test log_call_event convenience method."""
        event = default_context.log_call_event(
            call_id="CALL-001",
            event_type="CALL_STARTED",
            agent_id="AGENT-001",
            description="Inbound call started",
        )
        assert event.call_id == "CALL-001"
        assert event.event_category == "CALL"
        assert event.agent_id == "AGENT-001"

    def test_log_event_with_request_data(self, default_context):
        """Test logging event with request data."""
        event = default_context.log_event(
            event_type="API_CALL",
            event_category="INTEGRATION",
            description="CRM lookup",
            action="CRM_QUERY",
            request_data={"query": "customer_id:12345"},
        )
        assert event.request_data is not None
        assert event.request_data["query"] == "customer_id:12345"

    def test_log_event_tracks_tenant(self, tenant_context):
        """Test logged events include tenant ID."""
        event = tenant_context.log_event(
            event_type="TEST",
            event_category="TEST",
            description="Test",
            action="TEST",
        )
        assert event.tenant_id == "TENANT-A"

    def test_log_event_includes_frameworks(self, pci_context):
        """Test logged events include compliance frameworks."""
        event = pci_context.log_event(
            event_type="TEST",
            event_category="TEST",
            description="Test",
            action="TEST",
        )
        assert BPOComplianceFramework.PCI_DSS_4_0 in event.compliance_frameworks


# ---------------------------------------------------------------------------
# log_dtmf_masking_event()
# ---------------------------------------------------------------------------

class TestDTMFMaskingEvent:
    """Tests for log_dtmf_masking_event()."""

    def test_dtmf_masking_event(self, pci_context):
        """Test DTMF masking event is logged correctly."""
        event = pci_context.log_dtmf_masking_event(
            call_id="CALL-001",
            agent_id="AGENT-001",
            masking_mode="CLAMP",
        )
        assert event.event_type == "DTMF_MASKING"
        assert event.event_category == "PCI"
        assert event.call_id == "CALL-001"
        assert event.agent_id == "AGENT-001"
        assert event.resource_type == "VoiceChannel"

    def test_dtmf_masking_event_has_mode(self, pci_context):
        """Test DTMF masking event includes masking mode."""
        event = pci_context.log_dtmf_masking_event(
            call_id="CALL-002",
            agent_id="AGENT-002",
            masking_mode="FLAT",
        )
        assert event.request_data["masking_mode"] == "FLAT"

    @pytest.mark.parametrize("mode", ["CLAMP", "FLAT", "REPLACE"])
    def test_dtmf_masking_modes(self, pci_context, mode):
        """Test DTMF masking event with different modes."""
        event = pci_context.log_dtmf_masking_event(
            call_id="CALL-003",
            agent_id="AGENT-001",
            masking_mode=mode,
        )
        assert mode.lower() in event.description.lower() or mode in str(event.request_data)


# ---------------------------------------------------------------------------
# log_recording_pause_resume()
# ---------------------------------------------------------------------------

class TestRecordingPauseResume:
    """Tests for log_recording_pause_resume()."""

    def test_recording_pause(self, pci_context):
        """Test recording pause event."""
        event = pci_context.log_recording_pause_resume(
            call_id="CALL-001",
            agent_id="AGENT-001",
            action="PAUSE",
            reason="PCI_PAYMENT",
        )
        assert event.event_type == "RECORDING_PAUSE"
        assert event.event_category == "PCI"
        assert event.resource_type == "CallRecording"

    def test_recording_resume(self, pci_context):
        """Test recording resume event."""
        event = pci_context.log_recording_pause_resume(
            call_id="CALL-001",
            agent_id="AGENT-001",
            action="RESUME",
            reason="PCI_PAYMENT",
        )
        assert event.event_type == "RECORDING_RESUME"

    def test_pause_resume_sequence(self, pci_context):
        """Test a full pause/resume sequence."""
        pause_event = pci_context.log_recording_pause_resume(
            call_id="CALL-001",
            agent_id="AGENT-001",
            action="PAUSE",
        )
        resume_event = pci_context.log_recording_pause_resume(
            call_id="CALL-001",
            agent_id="AGENT-001",
            action="RESUME",
        )
        assert pause_event.timestamp <= resume_event.timestamp
        assert pause_event.call_id == resume_event.call_id

    def test_pause_resume_has_reason(self, pci_context):
        """Test pause/resume events include reason."""
        event = pci_context.log_recording_pause_resume(
            call_id="CALL-001",
            agent_id="AGENT-001",
            action="PAUSE",
            reason="PCI_PAYMENT",
        )
        assert event.request_data["reason"] == "PCI_PAYMENT"


# ---------------------------------------------------------------------------
# Audit Trail Querying
# ---------------------------------------------------------------------------

class TestAuditTrailQuerying:
    """Tests for audit trail querying."""

    def test_query_all_events(self, default_context):
        """Test querying all audit events."""
        default_context.log_event(
            event_type="EVENT_1",
            event_category="TEST",
            description="Event 1",
            action="ACTION_1",
        )
        default_context.log_event(
            event_type="EVENT_2",
            event_category="TEST",
            description="Event 2",
            action="ACTION_2",
        )
        events = default_context.get_audit_trail()
        assert len(events) >= 2

    def test_query_by_event_type(self, default_context):
        """Test querying by event type."""
        default_context.log_event(
            event_type="LOGIN",
            event_category="AUTH",
            description="Login",
            action="LOGIN",
        )
        default_context.log_event(
            event_type="LOGOUT",
            event_category="AUTH",
            description="Logout",
            action="LOGOUT",
        )
        events = default_context.get_audit_trail(event_type="LOGIN")
        assert all(e.event_type == "LOGIN" for e in events)

    def test_query_by_agent_id(self, default_context):
        """Test querying by agent ID."""
        default_context.log_event(
            event_type="ACTION",
            event_category="TEST",
            description="Agent 1 action",
            action="TEST",
            agent_id="AGENT-001",
        )
        default_context.log_event(
            event_type="ACTION",
            event_category="TEST",
            description="Agent 2 action",
            action="TEST",
            agent_id="AGENT-002",
        )
        events = default_context.get_audit_trail(agent_id="AGENT-001")
        assert all(e.agent_id == "AGENT-001" for e in events)
        assert len(events) >= 1

    def test_query_by_call_id(self, default_context):
        """Test querying by call ID."""
        default_context.log_call_event(
            call_id="CALL-100",
            event_type="CALL_START",
            description="Call started",
        )
        default_context.log_call_event(
            call_id="CALL-200",
            event_type="CALL_START",
            description="Different call",
        )
        events = default_context.get_audit_trail(call_id="CALL-100")
        assert all(e.call_id == "CALL-100" for e in events)

    def test_query_by_time_range(self, default_context):
        """Test querying by time range."""
        before = datetime.utcnow()
        default_context.log_event(
            event_type="TIMED_EVENT",
            event_category="TEST",
            description="Timed",
            action="TEST",
        )
        after = datetime.utcnow()
        events = default_context.get_audit_trail(
            start_time=before,
            end_time=after,
        )
        assert len(events) >= 1

    def test_query_with_limit(self, default_context):
        """Test querying with limit."""
        for i in range(10):
            default_context.log_event(
                event_type="BULK",
                event_category="TEST",
                description=f"Bulk event {i}",
                action="TEST",
            )
        events = default_context.get_audit_trail(limit=5)
        assert len(events) <= 5

    def test_empty_query_returns_empty(self, default_context):
        """Test querying empty audit trail."""
        events = default_context.get_audit_trail(event_type="NONEXISTENT")
        assert len(events) == 0


# ---------------------------------------------------------------------------
# verify_audit_chain()
# ---------------------------------------------------------------------------

class TestAuditChainIntegrity:
    """Tests for verify_audit_chain()."""

    def test_empty_chain_is_valid(self):
        """Test empty audit chain is valid."""
        context = BPOComplianceContext(frameworks=set())
        assert context.verify_audit_chain() is True

    def test_single_event_chain_valid(self, default_context):
        """Test single event chain is valid."""
        default_context.log_event(
            event_type="TEST",
            event_category="TEST",
            description="Single event",
            action="TEST",
        )
        assert default_context.verify_audit_chain() is True

    def test_multi_event_chain_valid(self, default_context):
        """Test multi-event chain maintains integrity."""
        for i in range(5):
            default_context.log_event(
                event_type="CHAIN_TEST",
                event_category="TEST",
                description=f"Chain event {i}",
                action="TEST",
            )
        assert default_context.verify_audit_chain() is True

    def test_tampered_chain_detected(self, default_context):
        """Test that tampering with audit chain is detected."""
        default_context.log_event(
            event_type="EVENT_1",
            event_category="TEST",
            description="First event",
            action="TEST",
        )
        default_context.log_event(
            event_type="EVENT_2",
            event_category="TEST",
            description="Second event",
            action="TEST",
        )

        # Tamper with the chain by modifying previous_hash
        if len(default_context._audit_trail) >= 2:
            default_context._audit_trail[1].previous_hash = "tampered_hash"
            assert default_context.verify_audit_chain() is False

    def test_chain_links_events_sequentially(self, default_context):
        """Test events are linked sequentially via hashes."""
        default_context.log_event(
            event_type="E1",
            event_category="TEST",
            description="Event 1",
            action="TEST",
        )
        default_context.log_event(
            event_type="E2",
            event_category="TEST",
            description="Event 2",
            action="TEST",
        )

        trail = default_context._audit_trail
        if len(trail) >= 2:
            # Second event's previous_hash should equal first event's hash
            assert trail[-1].previous_hash == trail[-2].event_hash


# ---------------------------------------------------------------------------
# get_compliance_summary()
# ---------------------------------------------------------------------------

class TestComplianceSummary:
    """Tests for get_compliance_summary()."""

    def test_summary_has_organization(self, default_context):
        """Test summary includes organization ID."""
        summary = default_context.get_compliance_summary()
        assert "organization_id" in summary

    def test_summary_has_domain(self, default_context):
        """Test summary includes domain name."""
        summary = default_context.get_compliance_summary()
        assert summary["domain"] == "BPO/Call Center"

    def test_summary_has_frameworks(self, default_context):
        """Test summary includes frameworks list."""
        summary = default_context.get_compliance_summary()
        assert "frameworks" in summary
        assert isinstance(summary["frameworks"], list)

    def test_summary_control_counts(self, pci_context):
        """Test summary has correct control counts."""
        summary = pci_context.get_compliance_summary()
        assert "controls" in summary
        assert summary["controls"]["total"] > 0
        total = (
            summary["controls"]["compliant"]
            + summary["controls"]["partially_compliant"]
            + summary["controls"]["non_compliant"]
            + summary["controls"]["under_review"]
        )
        assert total == summary["controls"]["total"]

    def test_summary_under_review_initially(self, pci_context):
        """Test all controls are under review initially."""
        summary = pci_context.get_compliance_summary()
        assert summary["controls"]["under_review"] == summary["controls"]["total"]
        assert summary["overall_status"] == "UNDER_REVIEW"

    def test_summary_overall_compliant(self, pci_context):
        """Test overall status becomes COMPLIANT when all controls pass."""
        # Mark all controls as compliant
        for control_id in list(pci_context._controls.keys()):
            pci_context.update_control_status(
                control_id=control_id,
                status=ComplianceStatus.COMPLIANT,
                assessed_by="auditor@test.com",
            )
        summary = pci_context.get_compliance_summary()
        assert summary["overall_status"] == "COMPLIANT"

    def test_summary_non_compliant_on_failure(self, pci_context):
        """Test overall status is NON_COMPLIANT when any control fails."""
        first_control_id = list(pci_context._controls.keys())[0]
        pci_context.update_control_status(
            control_id=first_control_id,
            status=ComplianceStatus.NON_COMPLIANT,
            assessed_by="auditor@test.com",
        )
        summary = pci_context.get_compliance_summary()
        assert summary["overall_status"] == "NON_COMPLIANT"

    def test_summary_by_framework(self, multi_framework_context):
        """Test summary groups controls by framework."""
        summary = multi_framework_context.get_compliance_summary()
        assert "by_framework" in summary
        assert len(summary["by_framework"]) > 0

    def test_summary_by_category(self, multi_framework_context):
        """Test summary groups controls by category."""
        summary = multi_framework_context.get_compliance_summary()
        assert "by_category" in summary
        assert len(summary["by_category"]) > 0

    def test_summary_has_assessment_date(self, default_context):
        """Test summary includes assessment date."""
        summary = default_context.get_compliance_summary()
        assert "assessment_date" in summary
        # Should be a valid ISO format date
        datetime.fromisoformat(summary["assessment_date"])


# ---------------------------------------------------------------------------
# Multi-Tenant Isolation
# ---------------------------------------------------------------------------

class TestMultiTenantIsolation:
    """Tests for multi-tenant compliance isolation."""

    def test_tenant_id_stored(self, tenant_context):
        """Test tenant ID is stored in context."""
        assert tenant_context.tenant_id == "TENANT-A"

    def test_tenant_events_tagged(self, tenant_context):
        """Test events are tagged with tenant ID."""
        event = tenant_context.log_event(
            event_type="TEST",
            event_category="TEST",
            description="Tenant event",
            action="TEST",
        )
        assert event.tenant_id == "TENANT-A"

    def test_different_tenants_independent(self):
        """Test different tenants have independent contexts."""
        context_a = BPOComplianceContext(
            frameworks={BPOComplianceFramework.PCI_DSS_4_0},
            tenant_id="TENANT-A",
        )
        context_b = BPOComplianceContext(
            frameworks={BPOComplianceFramework.PCI_DSS_4_0},
            tenant_id="TENANT-B",
        )

        context_a.log_event(
            event_type="A_EVENT",
            event_category="TEST",
            description="Tenant A event",
            action="TEST",
        )
        context_b.log_event(
            event_type="B_EVENT",
            event_category="TEST",
            description="Tenant B event",
            action="TEST",
        )

        a_events = context_a.get_audit_trail()
        b_events = context_b.get_audit_trail()

        assert all(e.tenant_id == "TENANT-A" for e in a_events)
        assert all(e.tenant_id == "TENANT-B" for e in b_events)

    def test_tenant_summary_isolation(self):
        """Test compliance summaries are tenant-specific."""
        context_a = BPOComplianceContext(
            frameworks={BPOComplianceFramework.PCI_DSS_4_0},
            tenant_id="TENANT-A",
        )
        context_b = BPOComplianceContext(
            frameworks={BPOComplianceFramework.HIPAA},
            tenant_id="TENANT-B",
        )

        summary_a = context_a.get_compliance_summary()
        summary_b = context_b.get_compliance_summary()

        assert summary_a["tenant_id"] == "TENANT-A"
        assert summary_b["tenant_id"] == "TENANT-B"


# ---------------------------------------------------------------------------
# Audit Event Serialization
# ---------------------------------------------------------------------------

class TestAuditEventSerialization:
    """Tests for BPOAuditEvent serialization."""

    def test_event_to_dict(self, default_context):
        """Test event serialization."""
        event = default_context.log_event(
            event_type="TEST",
            event_category="TEST",
            description="Serialization test",
            action="TEST",
            agent_id="AGENT-001",
            call_id="CALL-001",
        )
        d = event.to_dict()
        assert d["event_type"] == "TEST"
        assert d["agent_id"] == "AGENT-001"
        assert d["call_id"] == "CALL-001"
        assert "event_id" in d
        assert "timestamp" in d
        assert "event_hash" in d

    def test_event_hash_calculated(self, default_context):
        """Test event hash is calculated on creation."""
        event = default_context.log_event(
            event_type="TEST",
            event_category="TEST",
            description="Hash test",
            action="TEST",
        )
        assert event.event_hash is not None
        assert len(event.event_hash) > 0

    def test_control_to_dict(self, pci_context):
        """Test ComplianceControl serialization."""
        control = pci_context.get_control("PCI-CC-3.4.1")
        d = control.to_dict()
        assert d["control_id"] == "PCI-CC-3.4.1"
        assert "framework" in d
        assert "category" in d
        assert "status" in d

    def test_export_controls(self, pci_context):
        """Test exporting all controls."""
        controls = pci_context.export_controls()
        assert isinstance(controls, list)
        assert len(controls) > 0
        assert all("control_id" in c for c in controls)

    def test_export_audit_trail(self, default_context):
        """Test exporting audit trail."""
        default_context.log_event(
            event_type="EXPORT_TEST",
            event_category="TEST",
            description="Export test",
            action="TEST",
        )
        trail = default_context.export_audit_trail()
        assert isinstance(trail, list)
        assert len(trail) >= 1
        assert all("event_id" in e for e in trail)
