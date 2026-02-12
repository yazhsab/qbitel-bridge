"""
Integration tests for Container Security Pipeline.
Tests the flow from image scanning to admission control to runtime monitoring.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_native.container_security.image_scanning.vulnerability_scanner import VulnerabilityScanner, SeverityLevel
from cloud_native.container_security.admission_control.webhook_server import (
    AdmissionWebhookServer,
    SecurityPolicy,
    AdmissionAction,
)
from cloud_native.container_security.signing.dilithium_signer import DilithiumSigner
from cloud_native.container_security.runtime_protection.ebpf_monitor import eBPFMonitor, EventType


class TestContainerSecurityPipeline:
    """Test complete container security pipeline"""

    @pytest.mark.asyncio
    async def test_full_security_pipeline(self):
        """Test complete security pipeline: scan -> sign -> admit -> monitor"""
        image = "myapp:v1.0"

        # 1. Scan image
        scanner = VulnerabilityScanner(enable_quantum_scan=True)
        scan_results = await scanner._scan_quantum_vulnerabilities(image)

        # 2. Sign image if clean
        signer = DilithiumSigner(key_size=3)
        keypair = signer.generate_keypair()

        import hashlib

        image_digest = hashlib.sha256(image.encode()).hexdigest()
        signature = signer.sign_image(image_digest, keypair["private_key"])

        # 3. Admission control
        webhook = AdmissionWebhookServer(port=8443, tls_cert_path="/tmp/tls.crt", tls_key_path="/tmp/tls.key")

        policy = SecurityPolicy(
            name="production", allowed_registries=["docker.io"], require_signature=True, scan_for_quantum_vulnerabilities=True
        )

        webhook.add_policy(policy)

        pod = {"metadata": {"name": "test-pod"}, "spec": {"containers": [{"name": "app", "image": "docker.io/myapp:v1.0"}]}}

        admission_result = webhook.validate_pod(pod, "production")

        # 4. Runtime monitoring would start here
        events = []
        monitor = eBPFMonitor(event_callback=lambda e: events.append(e))

        assert scan_results is not None
        assert signature is not None
        assert admission_result is not None

    def test_image_signing_and_verification(self):
        """Test image signing and verification in admission control"""
        signer = DilithiumSigner(key_size=3)
        keypair = signer.generate_keypair()

        import hashlib

        image_digest = hashlib.sha256(b"test-image").hexdigest()

        # Sign
        signature = signer.sign_image(image_digest, keypair["private_key"])

        # Verify
        is_valid = signer.verify_signature(image_digest, signature["signature"], keypair["public_key"], signature["payload"])

        assert is_valid is True
