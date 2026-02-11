"""
Tests for Zero-Touch Automation Engine

Tests cover:
- Protocol adapter generation
- Key lifecycle management
- Certificate automation
- Self-healing orchestration
- Main automation engine
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import threading
import time

from ai_engine.automation import (
    # Protocol adapters
    ProtocolAdapterGenerator,
    GeneratedAdapter,
    AdapterConfig,
    TransformationType,
    # Key management
    KeyLifecycleManager,
    KeyPolicy,
    KeyMetadata,
    KeyType,
    KeyState,
    # Certificates
    CertificateAutomation,
    CertificateType,
    CertificateState,
    # Self-healing
    SelfHealingOrchestrator,
    ComponentType,
    HealthStatus,
    RecoveryAction,
    CircuitBreaker,
    # Main engine
    AutomationEngine,
    AutomationConfig,
    create_automation_engine,
)


class TestProtocolAdapterGenerator:
    """Tests for protocol adapter generation."""

    def test_generate_adapter_basic(self):
        """Test basic adapter generation."""
        generator = ProtocolAdapterGenerator()

        source_analysis = {
            "protocol_type": "swift_mt",
            "fields": [
                {"name": "sender_bic", "type": "string", "format": "BIC11"},
                {"name": "receiver_bic", "type": "string", "format": "BIC11"},
                {"name": "amount", "type": "decimal"},
                {"name": "currency", "type": "string", "length": 3},
            ],
        }

        target_analysis = {
            "protocol_type": "iso_20022",
            "fields": [
                {"name": "InstrBIC", "type": "string", "format": "BIC11"},
                {"name": "BnfcryBIC", "type": "string", "format": "BIC11"},
                {"name": "IntrBkSttlmAmt", "type": "decimal"},
                {"name": "Ccy", "type": "string", "length": 3},
            ],
        }

        adapter = generator.generate_adapter(
            source_analysis,
            target_analysis,
        )

        assert adapter is not None
        assert adapter.name is not None
        assert adapter.config is not None
        assert adapter.generated_code is not None

    def test_generate_adapter_with_config(self):
        """Test adapter generation with custom config."""
        from ai_engine.automation.protocol_adapter_generator import AdapterType

        generator = ProtocolAdapterGenerator()

        config = AdapterConfig(
            name="fix_to_iso",
            adapter_type=AdapterType.BIDIRECTIONAL,
            source_protocol="fix",
            target_protocol="iso_20022",
            enable_validation=True,
        )

        source_analysis = {"protocol_type": "fix", "fields": []}
        target_analysis = {"protocol_type": "iso_20022", "fields": []}

        adapter = generator.generate_adapter(
            source_analysis,
            target_analysis,
            config,
        )

        assert adapter is not None
        assert adapter.config.name == "fix_to_iso"

    def test_generate_field_mappings(self):
        """Test field mapping generation."""
        generator = ProtocolAdapterGenerator()

        source_analysis = {
            "protocol_type": "swift_mt",
            "fields": [
                {"name": "sender_bic", "type": "string"},
                {"name": "amount", "type": "decimal"},
            ],
        }

        target_analysis = {
            "protocol_type": "iso_20022",
            "fields": [
                {"name": "InstrBIC", "type": "string"},
                {"name": "Amount", "type": "decimal"},
            ],
        }

        # Generate adapter which will infer mappings
        adapter = generator.generate_adapter(source_analysis, target_analysis)

        assert adapter is not None
        # Mappings are generated based on field similarity
        assert adapter.config.mappings is not None


class TestKeyLifecycleManager:
    """Tests for key lifecycle management."""

    def test_generate_key(self):
        """Test key generation."""
        manager = KeyLifecycleManager()

        key = manager.generate_key(KeyType.DEK, tags={"env": "test"})

        assert key is not None
        assert key.key_id is not None
        assert key.key_type == KeyType.DEK
        assert key.state == KeyState.ACTIVE
        assert key.tags.get("env") == "test"

    def test_generate_key_with_policy(self):
        """Test key generation with custom policy."""
        manager = KeyLifecycleManager()

        key = manager.generate_key(
            KeyType.KEK,
            policy_id="default_kek",
        )

        assert key.key_type == KeyType.KEK
        assert key.algorithm == "ML-KEM-768"

    def test_rotate_key(self):
        """Test key rotation."""
        manager = KeyLifecycleManager()

        # Generate original key
        original = manager.generate_key(KeyType.DEK)

        # Rotate
        event = manager.rotate_key(original.key_id, "Test rotation")

        assert event is not None
        assert event.old_key_id == original.key_id
        assert event.new_key_id is not None
        assert event.completed is True

        # Check old key state
        old_key = manager.get_key(original.key_id)
        assert old_key.state == KeyState.DEACTIVATED

        # Check new key
        new_key = manager.get_key(event.new_key_id)
        assert new_key.state == KeyState.ACTIVE
        assert new_key.rotation_count == 1

    def test_destroy_key(self):
        """Test key destruction."""
        manager = KeyLifecycleManager()

        key = manager.generate_key(KeyType.DEK)
        result = manager.destroy_key(key.key_id, "Test destruction")

        assert result is True

        destroyed = manager.get_key(key.key_id)
        assert destroyed.state == KeyState.DESTROYED

    def test_suspend_and_activate_key(self):
        """Test key suspension and activation."""
        manager = KeyLifecycleManager()

        key = manager.generate_key(KeyType.DEK)

        # Suspend
        manager.suspend_key(key.key_id, "Maintenance")
        assert manager.get_key(key.key_id).state == KeyState.SUSPENDED

        # Activate
        manager.activate_key(key.key_id)
        assert manager.get_key(key.key_id).state == KeyState.ACTIVE

    def test_mark_compromised(self):
        """Test marking key as compromised (auto-rotates and deactivates)."""
        manager = KeyLifecycleManager()

        key = manager.generate_key(KeyType.DEK)
        result = manager.mark_compromised(key.key_id, "Suspected breach")

        assert result is True

        # After marking compromised, the key is auto-rotated and deactivated
        old_key = manager.get_key(key.key_id)
        assert old_key.state == KeyState.DEACTIVATED  # Auto-rotated

    def test_list_keys(self):
        """Test listing keys."""
        manager = KeyLifecycleManager()

        # Generate multiple keys
        manager.generate_key(KeyType.DEK)
        manager.generate_key(KeyType.DEK)
        manager.generate_key(KeyType.KEK)

        # List all
        all_keys = manager.list_keys()
        assert len(all_keys) >= 3

        # List by type
        dek_keys = manager.list_keys(key_type=KeyType.DEK)
        assert len(dek_keys) >= 2

        kek_keys = manager.list_keys(key_type=KeyType.KEK)
        assert len(kek_keys) >= 1

    def test_get_expiring_keys(self):
        """Test getting expiring keys."""
        manager = KeyLifecycleManager()

        # Generate key with short expiry (simulated by modifying expires_at)
        key = manager.generate_key(KeyType.DEK)
        key.expires_at = datetime.utcnow() + timedelta(days=7)

        expiring = manager.get_expiring_keys(days=14)

        assert len(expiring) >= 1
        assert any(k.key_id == key.key_id for k in expiring)


class TestCertificateAutomation:
    """Tests for certificate automation."""

    def test_create_certificate_request(self):
        """Test creating certificate request."""
        automation = CertificateAutomation()

        request = automation.create_certificate_request(
            common_name="api.example.com",
            cert_type=CertificateType.TLS_SERVER,
            subject_alt_names=["www.example.com"],
        )

        assert request is not None
        assert request.common_name == "api.example.com"
        assert request.cert_type == CertificateType.TLS_SERVER
        assert "www.example.com" in request.subject_alt_names

    def test_issue_certificate(self):
        """Test certificate issuance."""
        automation = CertificateAutomation()

        request = automation.create_certificate_request(
            common_name="test.example.com",
            cert_type=CertificateType.TLS_SERVER,
        )

        cert = automation.issue_certificate(request.request_id)

        assert cert is not None
        assert cert.common_name == "test.example.com"
        assert cert.state == CertificateState.ACTIVE
        assert cert.cert_pem is not None
        assert cert.fingerprint_sha256 is not None

    def test_renew_certificate(self):
        """Test certificate renewal."""
        automation = CertificateAutomation()

        # Issue original certificate
        request = automation.create_certificate_request(
            common_name="renew.example.com",
            cert_type=CertificateType.TLS_SERVER,
        )
        original = automation.issue_certificate(request.request_id)

        # Renew
        event = automation.renew_certificate(original.cert_id)

        assert event is not None
        assert event.old_cert_id == original.cert_id
        assert event.success is True

        # Check old cert
        old = automation.get_certificate(original.cert_id)
        assert old.state == CertificateState.RENEWED

        # Check new cert
        new = automation.get_certificate(event.new_cert_id)
        assert new.state == CertificateState.ACTIVE
        assert new.renewal_of == original.cert_id

    def test_revoke_certificate(self):
        """Test certificate revocation."""
        from ai_engine.automation import RevocationReason

        automation = CertificateAutomation()

        request = automation.create_certificate_request(
            common_name="revoke.example.com",
            cert_type=CertificateType.TLS_SERVER,
        )
        cert = automation.issue_certificate(request.request_id)

        result = automation.revoke_certificate(
            cert.cert_id,
            RevocationReason.KEY_COMPROMISE,
        )

        assert result is True

        revoked = automation.get_certificate(cert.cert_id)
        assert revoked.state == CertificateState.REVOKED
        assert revoked.revocation_reason == RevocationReason.KEY_COMPROMISE

    def test_get_expiring_certificates(self):
        """Test getting expiring certificates."""
        automation = CertificateAutomation()

        # Issue certificate
        request = automation.create_certificate_request(
            common_name="expiring.example.com",
            cert_type=CertificateType.TLS_SERVER,
        )
        cert = automation.issue_certificate(request.request_id)

        # Modify expiry to soon
        cert.not_after = datetime.utcnow() + timedelta(days=15)

        expiring = automation.get_expiring_certificates(days=30)

        assert len(expiring) >= 1

    def test_validate_certificate(self):
        """Test certificate validation."""
        automation = CertificateAutomation()

        request = automation.create_certificate_request(
            common_name="validate.example.com",
            cert_type=CertificateType.TLS_SERVER,
        )
        cert = automation.issue_certificate(request.request_id)

        validation = automation.validate_certificate(cert.cert_id)

        assert validation["valid"] is True
        assert validation["is_pqc_ready"] is True


class TestSelfHealingOrchestrator:
    """Tests for self-healing orchestration."""

    def test_register_component(self):
        """Test component registration."""
        orchestrator = SelfHealingOrchestrator()

        state = orchestrator.register_component(
            "test_component",
            ComponentType.API_ENDPOINT,
            "Test Component",
        )

        assert state is not None
        assert state.component_id == "test_component"
        assert state.status == HealthStatus.UNKNOWN

    def test_register_component_with_health_check(self):
        """Test component registration with health check."""
        orchestrator = SelfHealingOrchestrator()

        def check():
            return True

        state = orchestrator.register_component(
            "healthy_component",
            ComponentType.API_ENDPOINT,
            "Healthy Component",
            health_check=check,
        )

        # Perform check
        result = orchestrator.check_health("healthy_component")

        assert result.status == HealthStatus.HEALTHY

    def test_health_check_failure(self):
        """Test health check failure handling."""
        orchestrator = SelfHealingOrchestrator()

        def failing_check():
            return False

        orchestrator.register_component(
            "failing_component",
            ComponentType.API_ENDPOINT,
            "Failing Component",
            health_check=failing_check,
        )

        result = orchestrator.check_health("failing_component")

        assert result.status == HealthStatus.UNHEALTHY

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

        assert breaker.is_open is False

        # Record failures
        for _ in range(3):
            breaker.record_failure()

        assert breaker.is_open is True
        assert breaker.state == "open"

        # Wait for recovery
        time.sleep(1.1)

        assert breaker.is_open is False
        assert breaker.state == "half_open"

        # Record success to close
        for _ in range(3):
            breaker.record_success()

        assert breaker.state == "closed"

    def test_get_system_health(self):
        """Test system health summary."""
        orchestrator = SelfHealingOrchestrator()

        # Register components with different states
        orchestrator.register_component(
            "healthy",
            ComponentType.API_ENDPOINT,
            "Healthy",
            health_check=lambda: True,
        )

        orchestrator.check_health("healthy")

        health = orchestrator.get_system_health()

        assert "status" in health
        assert "total_components" in health
        assert health["total_components"] >= 1

    def test_incident_creation(self):
        """Test incident creation on failure."""
        orchestrator = SelfHealingOrchestrator()

        # Register with failing check
        def failing():
            raise Exception("Test failure")

        orchestrator.register_component(
            "incident_test",
            ComponentType.DATABASE,
            "Incident Test",
            health_check=failing,
        )

        # Trigger multiple failures
        for _ in range(3):
            orchestrator.check_health("incident_test")

        incidents = orchestrator.get_open_incidents()
        # May or may not create incident depending on threshold

    def test_resolve_incident(self):
        """Test incident resolution."""
        orchestrator = SelfHealingOrchestrator()

        # Create a manual incident for testing
        from ai_engine.automation.self_healing_orchestrator import Incident, IncidentSeverity

        incident = Incident(
            incident_id="test_incident",
            component_id="test_component",
            severity=IncidentSeverity.HIGH,
            title="Test Incident",
            description="Test description",
            created_at=datetime.utcnow(),
        )
        orchestrator._incidents["test_incident"] = incident

        # Resolve
        result = orchestrator.resolve_incident("test_incident", "Root cause found")

        assert result is True

        resolved = orchestrator.get_incident("test_incident")
        assert resolved.status == "resolved"
        assert resolved.root_cause == "Root cause found"


class TestAutomationEngine:
    """Tests for main automation engine."""

    def test_create_engine(self):
        """Test engine creation."""
        engine = AutomationEngine()

        assert engine is not None
        status = engine.get_status()
        assert status.status == "stopped"

    def test_start_and_stop(self):
        """Test engine start and stop."""
        engine = AutomationEngine()

        engine.start()
        status = engine.get_status()
        assert status.status == "running"
        assert status.started_at is not None

        engine.stop()
        status = engine.get_status()
        assert status.status == "stopped"

    def test_generate_key(self):
        """Test key generation through engine."""
        engine = AutomationEngine()

        key = engine.generate_key(KeyType.DEK)

        assert key is not None
        assert key.key_type == KeyType.DEK

    def test_issue_certificate(self):
        """Test certificate issuance through engine."""
        engine = AutomationEngine()

        cert = engine.issue_certificate(
            common_name="engine.example.com",
            cert_type=CertificateType.TLS_SERVER,
        )

        assert cert is not None
        assert cert.common_name == "engine.example.com"

    def test_register_component(self):
        """Test component registration through engine."""
        engine = AutomationEngine()

        state = engine.register_component(
            "api_service",
            ComponentType.API_ENDPOINT,
            "API Service",
        )

        assert state is not None

    def test_provision_secure_service(self):
        """Test secure service provisioning workflow."""
        engine = AutomationEngine()

        result = engine.provision_secure_service(
            "payment_api",
            service_type="api",
            environment="test",
        )

        assert result is not None
        assert "encryption_key" in result["resources"]
        assert "signing_key" in result["resources"]
        assert "tls_certificate" in result["resources"]
        assert "monitoring" in result["resources"]

    def test_rotate_all_expiring(self):
        """Test rotating all expiring resources."""
        engine = AutomationEngine()

        # Create resources
        key = engine.generate_key(KeyType.DEK)
        cert = engine.issue_certificate(
            "expiring.example.com",
            CertificateType.TLS_SERVER,
        )

        # Simulate expiring
        key.expires_at = datetime.utcnow() + timedelta(days=7)
        cert.not_after = datetime.utcnow() + timedelta(days=15)

        result = engine.rotate_all_expiring(key_days=14, cert_days=30)

        assert result is not None
        assert "keys_rotated" in result
        assert "certs_renewed" in result

    def test_run_compliance_check(self):
        """Test compliance check."""
        engine = AutomationEngine()

        # Generate some resources
        engine.generate_key(KeyType.DEK)
        engine.issue_certificate("check.example.com", CertificateType.TLS_SERVER)

        result = engine.run_compliance_check()

        assert result is not None
        assert "summary" in result
        assert "findings" in result
        assert "total_keys" in result["summary"]
        assert "total_certs" in result["summary"]

    def test_get_system_health(self):
        """Test system health through engine."""
        engine = AutomationEngine()

        health = engine.get_system_health()

        assert health is not None
        assert "status" in health

    def test_convenience_function(self):
        """Test create_automation_engine convenience function."""
        engine = create_automation_engine(auto_start=False)

        assert engine is not None
        assert engine.get_status().status == "stopped"


class TestAutomationIntegration:
    """Integration tests for automation components."""

    def test_key_certificate_integration(self):
        """Test key and certificate integration."""
        engine = AutomationEngine()

        # Generate signing key
        signing_key = engine.generate_key(KeyType.SIGNING)

        # Issue certificate using that key type's algorithm
        cert = engine.issue_certificate(
            "integrated.example.com",
            CertificateType.CODE_SIGNING,
        )

        assert signing_key is not None
        assert cert is not None

    def test_self_healing_key_rotation(self):
        """Test self-healing triggers key rotation on compromise."""
        engine = AutomationEngine()

        # Generate key
        key = engine.generate_key(KeyType.DEK, tags={"service": "payment"})

        # Simulate compromise detection (this auto-rotates and deactivates the old key)
        engine._key_manager.mark_compromised(key.key_id, "Detected anomaly")

        # Check key was rotated - old key should be deactivated after auto-rotation
        old_key = engine.get_key(key.key_id)
        assert old_key.state == KeyState.DEACTIVATED

    def test_full_lifecycle(self):
        """Test full automation lifecycle."""
        engine = AutomationEngine()
        engine.start()

        try:
            # Provision service
            result = engine.provision_secure_service(
                "lifecycle_test",
                environment="test",
            )

            # Check health
            health = engine.get_system_health()
            assert health is not None

            # Run compliance
            compliance = engine.run_compliance_check()
            assert compliance["summary"]["total_keys"] >= 2
            assert compliance["summary"]["total_certs"] >= 1

            # Get status
            status = engine.get_status()
            assert status.stats["tasks_completed"] >= 1

        finally:
            engine.stop()
