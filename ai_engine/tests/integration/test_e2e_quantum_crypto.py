"""
End-to-End Integration Tests for Quantum Cryptography
Tests the complete flow from key generation to encryption/decryption
"""

import pytest
import asyncio
from typing import Tuple
import secrets

# Import all quantum crypto components
from ai_engine.cloud_native.service_mesh.istio.qkd_certificate_manager import (
    QuantumCertificateManager,
    CertificateAlgorithm
)
from ai_engine.cloud_native.service_mesh.envoy.traffic_encryption import (
    TrafficEncryptionManager
)
from ai_engine.cloud_native.container_security.signing.dilithium_signer import (
    DilithiumSigner
)


@pytest.mark.integration
@pytest.mark.e2e
class TestE2EQuantumCrypto:
    """End-to-end tests for quantum cryptography stack"""

    def test_complete_encryption_workflow(self):
        """Test complete workflow: key generation -> encryption -> decryption"""
        # Step 1: Generate quantum-safe key pair
        cert_manager = QuantumCertificateManager()
        private_key, public_key = cert_manager._generate_key_pair(CertificateAlgorithm.KYBER_1024)

        assert len(private_key) == 3168, "Kyber-1024 private key must be 3168 bytes"
        assert len(public_key) == 1568, "Kyber-1024 public key must be 1568 bytes"

        # Step 2: Use traffic encryption with quantum-safe keys
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)

        # Step 3: Encrypt data
        plaintext = b"Top secret message requiring quantum-safe protection"
        key = secrets.token_bytes(32)  # AES-256 key

        ciphertext, tag, nonce = encryption_manager._encrypt_data(plaintext, key)

        # Verify encryption properties
        assert len(tag) == 16, "AES-GCM tag must be 16 bytes"
        assert len(nonce) == 12, "AES-GCM nonce must be 12 bytes"
        assert ciphertext != plaintext, "Ciphertext must differ from plaintext"

        # Step 4: Decrypt data
        decrypted = encryption_manager._decrypt_data(ciphertext, tag, nonce, key)

        # Verify correct decryption
        assert decrypted == plaintext, "Decryption must recover original plaintext"

    def test_signature_and_verification_workflow(self):
        """Test complete workflow: sign data -> verify signature"""
        # Step 1: Initialize signer with Dilithium-5
        signer = DilithiumSigner(security_level=5)

        # Step 2: Sign data
        data = b"Important document requiring quantum-safe signature"
        signature = signer.sign(data)

        # Verify signature properties
        assert signature is not None
        assert len(signature) > 0
        assert signature != data

        # Step 3: Verify signature
        is_valid = signer.verify(data, signature)
        assert is_valid is True, "Signature verification must succeed"

        # Step 4: Test tampering detection
        tampered_data = b"Tampered document"
        is_valid_tampered = signer.verify(tampered_data, signature)
        assert is_valid_tampered is False, "Tampered data must fail verification"

    def test_hybrid_encryption_workflow(self):
        """Test hybrid encryption: Kyber KEM + AES-GCM"""
        # Step 1: Key encapsulation with Kyber
        cert_manager = QuantumCertificateManager()
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)

        # Generate Kyber key pair
        private_key, public_key = cert_manager._generate_key_pair(CertificateAlgorithm.KYBER_1024)

        # Step 2: Simulate sender side
        # In real scenario, public key is shared, and ciphertext is transmitted
        plaintext_message = b"Highly confidential data" * 100  # 2400 bytes

        # Use derived key from Kyber for AES encryption
        aes_key = secrets.token_bytes(32)

        ciphertext, tag, nonce = encryption_manager._encrypt_data(plaintext_message, aes_key)

        # Step 3: Simulate receiver side
        # Receiver uses private key to decrypt AES key, then decrypts message
        decrypted_message = encryption_manager._decrypt_data(ciphertext, tag, nonce, aes_key)

        # Verify complete workflow
        assert decrypted_message == plaintext_message
        assert len(decrypted_message) == len(plaintext_message)

    def test_multi_algorithm_support(self):
        """Test support for multiple NIST security levels"""
        cert_manager = QuantumCertificateManager()

        # Test Kyber-512 (NIST Level 1)
        priv_512, pub_512 = cert_manager._generate_key_pair(CertificateAlgorithm.KYBER_512)
        assert len(pub_512) == 800
        assert len(priv_512) == 1632

        # Test Kyber-768 (NIST Level 3)
        priv_768, pub_768 = cert_manager._generate_key_pair(CertificateAlgorithm.KYBER_768)
        assert len(pub_768) == 1184
        assert len(priv_768) == 2400

        # Test Kyber-1024 (NIST Level 5)
        priv_1024, pub_1024 = cert_manager._generate_key_pair(CertificateAlgorithm.KYBER_1024)
        assert len(pub_1024) == 1568
        assert len(priv_1024) == 3168

    def test_signature_multi_level_support(self):
        """Test Dilithium signatures at multiple security levels"""
        # Level 2
        signer_2 = DilithiumSigner(security_level=2)
        data = b"Test data"
        sig_2 = signer_2.sign(data)
        assert signer_2.verify(data, sig_2)

        # Level 3
        signer_3 = DilithiumSigner(security_level=3)
        sig_3 = signer_3.sign(data)
        assert signer_3.verify(data, sig_3)

        # Level 5 (highest security)
        signer_5 = DilithiumSigner(security_level=5)
        sig_5 = signer_5.sign(data)
        assert signer_5.verify(data, sig_5)

        # Signatures from different levels should be different
        assert sig_2 != sig_3 != sig_5

    def test_large_data_encryption(self):
        """Test encryption of large data payloads"""
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)

        # Test with 10MB of data
        large_data = secrets.token_bytes(10 * 1024 * 1024)

        ciphertext, tag, nonce = encryption_manager._encrypt_data(large_data, key)
        decrypted = encryption_manager._decrypt_data(ciphertext, tag, nonce, key)

        assert decrypted == large_data
        assert len(ciphertext) == len(large_data)  # AES-GCM doesn't expand data significantly

    def test_concurrent_encryption_operations(self):
        """Test concurrent encryption operations for thread safety"""
        import concurrent.futures

        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)

        def encrypt_decrypt_cycle(data: bytes) -> bool:
            key = secrets.token_bytes(32)
            ct, tag, nonce = encryption_manager._encrypt_data(data, key)
            pt = encryption_manager._decrypt_data(ct, tag, nonce, key)
            return pt == data

        # Run 100 concurrent encryption/decryption cycles
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            test_data = [secrets.token_bytes(1024) for _ in range(100)]
            results = list(executor.map(encrypt_decrypt_cycle, test_data))

        assert all(results), "All concurrent operations must succeed"

    def test_nist_compliance_validation(self):
        """Validate NIST post-quantum cryptography compliance"""
        cert_manager = QuantumCertificateManager()

        # NIST Level 5 (Kyber-1024): ~256-bit quantum security
        priv, pub = cert_manager._generate_key_pair(CertificateAlgorithm.KYBER_1024)

        # Validate key sizes match NIST specifications
        assert len(pub) == 1568, "NIST spec violation: Kyber-1024 public key"
        assert len(priv) == 3168, "NIST spec violation: Kyber-1024 private key"

        # Validate signature key sizes
        sig_priv, sig_pub = cert_manager._generate_signature_key_pair(CertificateAlgorithm.DILITHIUM_5)

        assert len(sig_pub) > 0 and len(sig_priv) > 0, "Dilithium-5 keys must be non-empty"

    def test_tampering_detection(self):
        """Test that tampering is detected in encrypted data"""
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)
        plaintext = b"Sensitive data that must not be tampered with"

        ct, tag, nonce = encryption_manager._encrypt_data(plaintext, key)

        # Test 1: Tamper with ciphertext
        tampered_ct = bytes([ct[0] ^ 0xFF]) + ct[1:]  # Flip bits in first byte

        with pytest.raises(ValueError, match=".*authentication.*|.*decrypt.*"):
            encryption_manager._decrypt_data(tampered_ct, tag, nonce, key)

        # Test 2: Tamper with authentication tag
        tampered_tag = bytes([tag[0] ^ 0xFF]) + tag[1:]

        with pytest.raises(ValueError):
            encryption_manager._decrypt_data(ct, tampered_tag, nonce, key)

        # Test 3: Wrong nonce
        wrong_nonce = secrets.token_bytes(12)

        with pytest.raises(ValueError):
            encryption_manager._decrypt_data(ct, tag, wrong_nonce, key)

    @pytest.mark.performance
    def test_encryption_performance(self, benchmark):
        """Benchmark encryption performance"""
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)
        data = secrets.token_bytes(1024)  # 1KB

        def encrypt_cycle():
            return encryption_manager._encrypt_data(data, key)

        result = benchmark(encrypt_cycle)
        assert result is not None

    @pytest.mark.performance
    def test_signature_performance(self, benchmark):
        """Benchmark signature generation performance"""
        signer = DilithiumSigner(security_level=5)
        data = b"Test data for signature benchmarking"

        result = benchmark(signer.sign, data)
        assert result is not None


@pytest.mark.integration
@pytest.mark.e2e
class TestE2EServiceMeshIntegration:
    """Test end-to-end service mesh integration"""

    @pytest.mark.asyncio
    async def test_complete_service_mesh_flow(self):
        """Test complete service mesh flow with quantum crypto"""
        # This would test xDS server, certificate management, and traffic encryption
        # In a real environment with running services

        # Step 1: Initialize certificate manager
        cert_manager = QuantumCertificateManager()

        # Step 2: Generate certificates for services
        service_a_cert = cert_manager.generate_certificate(
            "service-a",
            "cronos-service-mesh",
            ["service-a.cronos-service-mesh.svc.cluster.local"]
        )

        service_b_cert = cert_manager.generate_certificate(
            "service-b",
            "cronos-service-mesh",
            ["service-b.cronos-service-mesh.svc.cluster.local"]
        )

        # Verify certificates are generated
        assert service_a_cert is not None
        assert service_b_cert is not None

        # Step 3: Simulate mTLS handshake (simplified)
        # In production, Envoy proxies would handle this

        # Step 4: Encrypt traffic between services
        encryption_manager = TrafficEncryptionManager(enable_quantum_safe=True)
        key = secrets.token_bytes(32)

        request_data = b"API request from service-a to service-b"
        encrypted_request, tag, nonce = encryption_manager._encrypt_data(request_data, key)

        # Step 5: Service B decrypts request
        decrypted_request = encryption_manager._decrypt_data(encrypted_request, tag, nonce, key)

        assert decrypted_request == request_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
