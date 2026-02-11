# QBITEL - Banking Domain Product Walkthrough

## Product Definition

**QBITEL** is an **enterprise-grade, quantum-safe security platform** that provides:

- **AI-Powered Protocol Discovery** - Automatically discover, analyze, and understand legacy banking protocols
- **Cloud-Native Security** - Comprehensive security for containerized and cloud-deployed banking workloads
- **Post-Quantum Cryptography** - NIST-approved PQC algorithms to protect against quantum computing threats
- **Legacy System Protection** - Secure legacy mainframe and midrange systems without full replacement
- **Zero-Touch Automation** - Fully automated key lifecycle, certificate management, self-healing, and policy enforcement

The platform protects both **legacy systems** and **modern infrastructure** against current and future quantum computing threats with **minimal human intervention**.

---

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [Core Capabilities](#core-capabilities)
3. [Zero-Touch Automation](#zero-touch-automation)
4. [Banking Use Cases](#banking-use-cases)
5. [Protocol Support](#protocol-support)
6. [Security Architecture](#security-architecture)
7. [Compliance Frameworks](#compliance-frameworks)
8. [Getting Started](#getting-started)
9. [API Reference](#api-reference)

---

## Platform Overview

### The Quantum Threat to Banking

By 2030-2035, quantum computers are expected to break current cryptographic standards:
- **RSA-2048**: Vulnerable to Shor's algorithm
- **ECDSA P-256**: Vulnerable to quantum attacks
- **3DES/AES key exchange**: Vulnerable without PQC

Banks must start migrating NOW to protect:
- Wire transfers and payment messages
- HSM-stored cryptographic keys
- Digital signatures on transactions
- Customer authentication systems

### QBITEL Solution

| Challenge | QBITEL Solution |
|-----------|-------------------|
| Unknown legacy protocols | AI-powered protocol discovery |
| Cryptographic migration | Hybrid PQC with gradual rollout |
| Cloud security gaps | Cloud-native security controls |
| Compliance requirements | Built-in regulatory frameworks |
| HSM vendor lock-in | Multi-vendor HSM abstraction |
| Manual key rotation | Zero-touch automated key lifecycle |
| Certificate expiration | Auto-renewal with policy-driven issuance |
| System failures | Self-healing with automated recovery |

### Key Differentiators

| Capability | Traditional Approach | QBITEL |
|------------|---------------------|-----------|
| Protocol Analysis | Manual reverse engineering | AI-powered auto-discovery |
| PQC Migration | Years-long manual effort | Automated hybrid deployment |
| HSM Integration | Single-vendor lock-in | Multi-cloud HSM abstraction |
| Threat Detection | Signature-based | AI/ML anomaly detection |
| Compliance | Point-in-time audits | Continuous monitoring |
| Key Management | Manual rotation schedules | Zero-touch auto-rotation |
| Certificate Lifecycle | Manual renewal tickets | Policy-driven auto-renewal |
| Incident Response | Manual remediation | Self-healing automation |
| Security Configuration | Manual hardening | Auto-configuration |

---

## Core Capabilities

### 1. AI-Powered Protocol Discovery

Automatically discover and analyze legacy banking protocols without source code access:

```python
from ai_engine.discovery.protocol_discovery_orchestrator import (
    ProtocolDiscoveryOrchestrator,
    DiscoveryConfig,
)

# Initialize the discovery engine
orchestrator = ProtocolDiscoveryOrchestrator()

# Discover protocols from network traffic or message samples
result = await orchestrator.discover_protocols(
    source="network_capture",
    data_path="/captures/legacy_mainframe.pcap",
    config=DiscoveryConfig(
        enable_ml_classification=True,
        grammar_inference=True,
        generate_parser=True,
    ),
)

# Results include:
# - Protocol identification (SWIFT MT, ISO 8583, FIX, etc.)
# - Field structure analysis
# - Grammar rules for parsing
# - Generated parser code
for protocol in result.discovered_protocols:
    print(f"Discovered: {protocol.name}")
    print(f"  Confidence: {protocol.confidence}%")
    print(f"  Fields: {len(protocol.fields)}")
    print(f"  Parser: {protocol.generated_parser_path}")
```

**Capabilities**:
- Statistical pattern recognition
- PCFG (Probabilistic Context-Free Grammar) inference
- Field boundary detection
- Semantic type classification
- Auto-generated parsers and validators

---

### 2. Post-Quantum Cryptography

Protect banking systems against quantum computing threats with NIST-approved algorithms:

```python
from ai_engine.domains.banking.security.pqc import (
    MLKEM768,
    MLDSA65,
    HybridKeyExchange,
)
from ai_engine.domains.banking.core.domain_profile import (
    BankingPQCProfile,
    BankingSubdomain,
)

# Select optimized PQC profile for wire transfers
profile = BankingPQCProfile.for_subdomain(BankingSubdomain.WIRE_TRANSFERS)
# Auto-selects: ML-KEM-768 + ML-DSA-65 (192-bit security)

# Hybrid key exchange (classical + PQC)
hybrid = HybridKeyExchange(
    classical="X25519",      # Current standard
    pqc="ML-KEM-768",        # NIST PQC standard
    mode="HYBRID",           # Both must validate
)

# Encapsulate shared secret
ciphertext, shared_secret = hybrid.encapsulate(recipient_public_key)

# Sign with hybrid signature
signature = hybrid.sign(
    message=wire_transfer_data,
    algorithm="Ed25519-ML-DSA-65",  # Hybrid signature
)
```

**Supported Algorithms** (NIST FIPS 203/204):

| Algorithm | Type | Security Level | Use Case |
|-----------|------|----------------|----------|
| ML-KEM-512 | Key Encapsulation | 128-bit | Real-time payments |
| ML-KEM-768 | Key Encapsulation | 192-bit | Wire transfers |
| ML-KEM-1024 | Key Encapsulation | 256-bit | HSM master keys |
| ML-DSA-44 | Digital Signature | 128-bit | High-frequency trading |
| ML-DSA-65 | Digital Signature | 192-bit | Payment signing |
| ML-DSA-87 | Digital Signature | 256-bit | Root CA certificates |

---

### 3. Cloud-Native Security

Comprehensive security controls for banking workloads in cloud environments:

```python
from ai_engine.cloud_native.container_security import (
    ImageScanner,
    AdmissionController,
    RuntimeProtector,
    ContainerSigner,
)
from ai_engine.cloud_native.service_mesh import (
    IstioMTLSConfig,
    AuthorizationPolicy,
)

# Container image security
scanner = ImageScanner(
    vulnerability_threshold="HIGH",
    compliance_standards=["PCI-DSS", "CIS-Benchmark"],
)

# Scan and sign with PQC
scan_result = await scanner.scan_image("payments-service:v2.0")
if scan_result.is_secure:
    # Sign with Dilithium (ML-DSA)
    signer = ContainerSigner(algorithm="ML-DSA-65")
    signed = await signer.sign_image("payments-service:v2.0")

# Service mesh mTLS with PQC
mtls_config = IstioMTLSConfig(
    mode="STRICT",
    min_tls_version="TLSv1.3",
    pqc_enabled=True,  # Hybrid TLS with ML-KEM
)

# Runtime protection with eBPF
protector = RuntimeProtector(
    threat_detection=True,
    file_integrity_monitoring=True,
    network_policy_enforcement=True,
)
```

**Security Controls**:
- Container image vulnerability scanning
- PQC-signed container images (ML-DSA)
- Kubernetes admission policies
- Istio mTLS with hybrid PQC
- eBPF runtime threat detection
- Network segmentation policies

---

### 4. Legacy System Protection

Secure legacy mainframe and midrange systems without replacement:

```python
from ai_engine.services.legacy_whisperer import (
    LegacyWhisperer,
    ProtocolAdapter,
)
from ai_engine.domains.banking.security.hsm import HSMProvider

# Connect to legacy system securely
whisperer = LegacyWhisperer(
    legacy_endpoint="cics://mainframe.bank.internal:1490",
    security={
        "encryption": "TLS1.3",
        "pqc_tunnel": True,  # Quantum-safe tunnel
    },
)

# Discover and wrap legacy transactions
transactions = await whisperer.discover_transactions()
for txn in transactions:
    # Generate secure API wrapper
    adapter = await whisperer.generate_adapter(
        transaction=txn,
        target="REST",
        security_controls={
            "authentication": "OAuth2",
            "encryption": "AES-256-GCM",
            "signing": "ML-DSA-65",
        },
    )

# Protect legacy HSM keys with PQC wrapping
legacy_hsm = HSMProvider.create("thales_luna")
cloud_hsm = HSMProvider.create("aws_cloudhsm", pqc_enabled=True)

# Wrap legacy keys with PQC for quantum-safe storage
for key in await legacy_hsm.list_keys():
    wrapped = await cloud_hsm.wrap_key_pqc(
        key_material=key,
        wrapping_algorithm="ML-KEM-768",
    )
    await secure_backup.store(wrapped)
```

**Legacy Protection Features**:
- Quantum-safe tunnels to legacy systems
- API wrappers with modern security
- PQC key wrapping for legacy HSM keys
- Protocol translation with security upgrade
- Zero-downtime security enhancement

---

## Zero-Touch Automation

QBITEL provides **fully automated security operations** that eliminate manual intervention for routine security tasks while maintaining complete audit trails and policy compliance.

### 5. Zero-Touch Automation Engine

The automation engine combines five core components for hands-free security operations:

```python
from ai_engine.automation import (
    AutomationEngine,
    AutomationConfig,
    KeyLifecycleManager,
    CertificateAutomation,
    SelfHealingOrchestrator,
    ConfigurationManager,
    ProtocolAdapterGenerator,
)
from ai_engine.domains.banking.core.security_policy import BankingSecurityPolicy

# Configure zero-touch automation
config = AutomationConfig(
    key_rotation_interval=3600,      # Check every hour
    key_rotation_days=90,            # Rotate keys every 90 days
    cert_renewal_interval=86400,     # Check daily
    cert_renewal_threshold_days=30,  # Renew 30 days before expiry
    enable_self_healing=True,
    enable_auto_configuration=True,
)

# Initialize the automation engine
engine = AutomationEngine(config=config)

# Start zero-touch operations
await engine.start()

# Engine now automatically:
# - Rotates expiring keys (PQC and classical)
# - Renews certificates before expiration
# - Recovers from HSM/database failures
# - Auto-configures security settings
# - Generates protocol adapters on demand
```

---

### 5.1 Automated Key Lifecycle Management

Zero-touch key management with PQC support, automatic rotation, and compromise handling:

```python
from ai_engine.automation import KeyLifecycleManager, KeyType, KeyState
from ai_engine.domains.banking.security.hsm import HSMProvider

# Initialize with HSM
hsm = HSMProvider.create("aws_cloudhsm", pqc_enabled=True)
key_manager = KeyLifecycleManager(hsm=hsm)

# Start automatic key rotation scheduler
key_manager.start_rotation_scheduler()

# Generate PQC key with auto-rotation policy
key = await key_manager.generate_key(
    key_type=KeyType.KEK,           # Key Encryption Key
    algorithm="ML-KEM-768",         # PQC algorithm
    rotation_days=365,              # Auto-rotate annually
    requires_hsm=True,              # HSM-backed
    dual_control=True,              # Requires two approvers
)

# Register rotation callback for external notifications
key_manager.register_rotation_callback(
    callback=notify_key_management_system,
    key_types=[KeyType.KEK, KeyType.DEK],
)

# Automatic compromise handling
await key_manager.mark_compromised(
    key_id=compromised_key_id,
    reason="Potential exposure detected",
)
# Automatically triggers immediate rotation and incident logging
```

**Automated Key Policies (Banking)**:

| Key Type | Algorithm | Rotation | HSM Required | Use Case |
|----------|-----------|----------|--------------|----------|
| KEK (Master) | ML-KEM-1024 | 365 days | Yes | Key encryption |
| DEK (Data) | AES-256 | 90 days | No | Data encryption |
| Signing | ML-DSA-65 | 365 days | Yes | Transaction signing |
| Transport | ML-KEM-768 | 90 days | Yes | Key transport |

---

### 5.2 Automated Certificate Lifecycle

Policy-driven certificate issuance, auto-renewal, and revocation:

```python
from ai_engine.automation import CertificateAutomation, CertificateType, CertPolicy

# Initialize certificate automation
cert_auto = CertificateAutomation(
    key_manager=key_manager,
    ca_provider=internal_ca,
)

# Start auto-renewal scheduler
cert_auto.start_renewal_scheduler()

# Define certificate policy
policy = CertPolicy(
    cert_type=CertificateType.TLS_SERVER,
    validity_days=90,
    auto_renew=True,
    renewal_threshold_days=30,
    allowed_domains=["*.bank.internal", "*.payments.bank.com"],
    require_approval=False,  # Auto-approve for internal certs
    ocsp_must_staple=True,
    pqc_ready=True,  # Use hybrid signatures
)

# Issue certificate (zero-touch for policy-approved requests)
cert = await cert_auto.issue_certificate(
    common_name="payments-api.bank.internal",
    san=["payments-api", "payments-api.bank.internal"],
    policy=policy,
)

# Certificate auto-renews 30 days before expiration
# No manual intervention required
```

**Certificate States (Automatic Transitions)**:

```
PENDING → ACTIVE → EXPIRING → RENEWED
                ↘ REVOKED (on compromise)
                ↘ EXPIRED (if renewal fails)
```

---

### 5.3 Self-Healing Orchestrator

Automated failure detection and recovery without human intervention:

```python
from ai_engine.automation import (
    SelfHealingOrchestrator,
    ComponentType,
    RecoveryAction,
    RecoveryPlan,
)

# Initialize self-healing
healer = SelfHealingOrchestrator()

# Register components for health monitoring
healer.register_component(
    name="primary-hsm",
    component_type=ComponentType.HSM,
    health_check=hsm_health_check_function,
    check_interval_seconds=30,
    failure_threshold=5,  # 5 failures = trigger recovery
)

healer.register_component(
    name="banking-database",
    component_type=ComponentType.DATABASE,
    health_check=db_health_check_function,
    check_interval_seconds=10,
    failure_threshold=3,
)

# Define recovery plan
hsm_recovery = RecoveryPlan(
    component_type=ComponentType.HSM,
    actions=[
        RecoveryAction.RECONNECT,       # 1st: Try reconnect
        RecoveryAction.FAILOVER,        # 2nd: Failover to backup
        RecoveryAction.CIRCUIT_BREAK,   # 3rd: Circuit break
        RecoveryAction.ALERT_ONLY,      # 4th: Alert for manual
    ],
    max_attempts=3,
    cooldown_seconds=60,
)

healer.set_recovery_plan(ComponentType.HSM, hsm_recovery)

# Start health monitoring
await healer.start()

# Automatic recovery on failure:
# 1. Detects HSM health check failure
# 2. Opens circuit breaker after 5 failures
# 3. Executes recovery: RECONNECT → FAILOVER → CIRCUIT_BREAK
# 4. Logs incident with full audit trail
# 5. Alerts if manual intervention needed
```

**Default Recovery Plans (Banking)**:

| Component | Recovery Actions | Attempts | Cooldown |
|-----------|-----------------|----------|----------|
| HSM | Reconnect → Failover → Circuit Break | 3 | 60s |
| Database | Reconnect → Failover | 5 | 30s |
| API Endpoint | Circuit Break → Alert | 3 | 120s |
| Crypto Provider | Reconnect → Rotate Credentials → Failover | 3 | 60s |

---

### 5.4 Auto-Configuration Manager

Automatic security configuration based on environment and compliance requirements:

```python
from ai_engine.automation import ConfigurationManager, ConfigSource

# Initialize configuration manager
config_mgr = ConfigurationManager()

# Auto-detect environment and configure
security_config = await config_mgr.auto_configure(
    environment="production",
    compliance_frameworks=["PCI-DSS", "DORA", "NIST-PQC"],
)

# Auto-detection features:
# - Cloud provider (AWS/Azure/GCP) from environment
# - Compliance frameworks from protocols detected
# - PQC settings based on security score
# - HSM provider based on cloud environment

print(f"Auto-configured:")
print(f"  PQC Mode: {security_config.pqc.hybrid_mode}")  # True
print(f"  HSM Provider: {security_config.hsm.provider}")  # aws_cloudhsm
print(f"  Key Rotation: {security_config.encryption.key_rotation_days} days")
print(f"  Compliance: {security_config.compliance.frameworks}")
```

**Auto-Configuration Rules**:

| Detection | Configuration Action |
|-----------|---------------------|
| AWS environment | Use AWS CloudHSM |
| SWIFT protocol detected | Enable SWIFT-CSP compliance |
| EMV protocol detected | Enable PCI-DSS compliance |
| Security score < 70 | Shorten key rotation to 30 days |
| PQC migration enabled | Enable hybrid cryptography mode |

---

### 5.5 Zero-Touch Protocol Adapter Generation

AI-powered automatic generation of protocol adapters with validation and tests:

```python
from ai_engine.automation import ProtocolAdapterGenerator
from ai_engine.discovery import DiscoveredProtocol

# Initialize adapter generator
generator = ProtocolAdapterGenerator()

# Generate adapter from discovered protocol
adapter = await generator.generate_adapter(
    source_protocol=discovered_legacy_protocol,
    target_format="internal_banking_format",
)

# Auto-generates:
# - Python adapter class with transform/reverse_transform
# - Field mappings with confidence scores
# - Validation logic for required fields
# - Pytest test cases
# - Markdown documentation

print(f"Generated adapter: {adapter.class_name}")
print(f"Complexity score: {adapter.complexity_score}")
print(f"Field mappings: {len(adapter.field_mappings)}")
print(f"Test file: {adapter.test_file_path}")
print(f"Documentation: {adapter.docs_path}")
```

---

### 5.6 Banking-Specific Security Policies

Pre-defined zero-touch policies for banking operations:

```python
from ai_engine.domains.banking.core.security_policy import BankingSecurityPolicy

# Payment Processing Policy (auto-enforced)
payment_policy = BankingSecurityPolicy.for_payment_processing()
# Auto-enforces:
# - ML-KEM-768, ML-DSA-65 encryption/signing
# - Dual control required
# - 4-hour session limit
# - PAN masking in logs
# - 7-year audit retention

# Wire Transfer Policy (highest security)
wire_policy = BankingSecurityPolicy.for_wire_transfer()
# Auto-enforces:
# - ML-KEM-1024, ML-DSA-87 (highest PQC security)
# - Four-Eyes Principle
# - Rate limiting: 100/day, $10M/day
# - 15-minute incident response SLA
# - 10-year audit retention

# Trading Systems Policy (low latency)
trading_policy = BankingSecurityPolicy.for_trading()
# Auto-enforces:
# - ML-KEM-512, Falcon-512 (fastest PQC)
# - MiFID II + EMIR compliance
# - 30-minute idle timeout
# - 7-year retention

# HSM Operations Policy (maximum security)
hsm_policy = BankingSecurityPolicy.for_hsm_operations()
# Auto-enforces:
# - Hardware token MFA only
# - Dual control
# - 1-hour session limit
# - Real-time alerting
# - 5-minute incident response SLA
```

---

### 5.7 Zero-Touch Service Provisioning

Complete service deployment with security in one call:

```python
from ai_engine.automation import AutomationEngine
from ai_engine.domains.banking.core.security_policy import BankingSecurityPolicy

engine = AutomationEngine(config=config)

# Provision a new secure service (zero-touch)
service = await engine.provision_secure_service(
    service_name="wire-transfer-api",
    policy=BankingSecurityPolicy.for_wire_transfer(),
)

# Automatically:
# 1. Generates DEK (AES-256-GCM) for data encryption
# 2. Generates signing key (ML-DSA-65) for transactions
# 3. Issues TLS certificate with SANs
# 4. Configures security settings per policy
# 5. Registers for health monitoring
# 6. Enables compliance auditing

print(f"Service provisioned: {service.name}")
print(f"  Encryption key: {service.encryption_key_id}")
print(f"  Signing key: {service.signing_key_id}")
print(f"  TLS cert: {service.certificate_id}")
print(f"  Health monitoring: Enabled")
print(f"  Compliance: {service.compliance_frameworks}")
```

---

### Zero-Touch Automation Summary

| Component | Automation | Interval | Banking Benefit |
|-----------|------------|----------|-----------------|
| **Key Lifecycle** | Auto-rotation, compromise handling | Hourly check | No key expiration incidents |
| **Certificate** | Auto-renewal, policy issuance | Daily check | No certificate outages |
| **Self-Healing** | Auto-recovery, circuit breakers | 30s per component | 99.99% availability |
| **Configuration** | Auto-detection, compliance mapping | On-demand | Instant compliance |
| **Adapters** | Code generation with tests | On-demand | Rapid integration |
| **Policies** | Pre-defined banking security | Per-operation | Consistent security |

**Key Benefits**:
- **Eliminate Human Error**: Automated rotation prevents missed key/cert renewals
- **Faster Incident Response**: Self-healing recovers in seconds, not hours
- **Continuous Compliance**: Policy-driven automation ensures regulatory adherence
- **Reduced Operations Cost**: Zero manual intervention for routine security tasks
- **Complete Audit Trail**: Every automated action is logged with full context

---

## Banking Use Cases

### Use Case 1: Wire Transfer Security (FedWire/SWIFT)

**Threat**: Quantum computers could forge wire transfer signatures or decrypt intercepted messages.

**Solution**: Hybrid PQC signatures and encryption for wire transfers.

```python
from ai_engine.domains.banking.protocols.payments.domestic.fedwire import (
    FedWireBuilder,
    FedWireMessage,
)
from ai_engine.domains.banking.security.hsm import HSMProvider

# Initialize HSM with PQC support
hsm = HSMProvider.create("aws_cloudhsm", pqc_enabled=True)

# Generate hybrid signing key
signing_key = await hsm.generate_key_pair(
    algorithm="Ed25519-ML-DSA-65",  # Hybrid: classical + PQC
    purpose="wire_transfer_signing",
)

# Build wire transfer
builder = FedWireBuilder()
wire = builder.create_funds_transfer(
    sender_routing="021000089",
    receiver_routing="021000021",
    amount=5_000_000.00,
    business_function="CTR",
)

# Sign with quantum-safe hybrid signature
signature = await hsm.sign(
    data=wire.to_wire_format(),
    key=signing_key,
    algorithm="Ed25519-ML-DSA-65",
)

# Encrypt with hybrid key exchange
encrypted = await hsm.encrypt(
    plaintext=wire.to_wire_format(),
    algorithm="X25519-ML-KEM-768",  # Hybrid encryption
    recipient_key=recipient_public_key,
)
```

**Protection**:
- Signatures valid against both classical and quantum attacks
- Encrypted messages protected for 20+ years
- HSM-backed key management

---

### Use Case 2: EMV Card Transaction Verification

**Threat**: Quantum computers could forge EMV cryptograms or derive session keys.

**Solution**: HSM-backed verification with PQC-ready key hierarchy.

```python
from ai_engine.domains.banking.security.crypto.verification import (
    EMVCryptogramVerifier,
    EMVCryptogramData,
    CryptogramType,
)
from ai_engine.domains.banking.security.hsm import HSMProvider

# Initialize verifier with HSM
hsm = HSMProvider.create("thales_luna")
verifier = EMVCryptogramVerifier(hsm=hsm)

# Verify ARQC from card transaction
data = EMVCryptogramData(
    cryptogram=bytes.fromhex("A1B2C3D4E5F60718"),
    cryptogram_type=CryptogramType.ARQC,
    pan="4761111111111111",
    pan_sequence="00",
    atc=bytes.fromhex("0001"),
    amount_authorized=10000,
    terminal_country_code="0840",
    transaction_currency_code="0840",
    transaction_date="250115",
    transaction_type="00",
    unpredictable_number=bytes.fromhex("12345678"),
)

# Verify with HSM (keys never leave HSM)
result = verifier.verify_arqc(data, issuer_master_key_handle)

if result.status == VerificationStatus.VALID:
    # Generate ARPC response
    arpc, _ = verifier.generate_arpc(
        arqc=data.cryptogram,
        authorization_response_code="3030",  # Approved
        issuer_master_key=issuer_master_key_handle,
        pan=data.pan,
        atc=data.atc,
    )
```

**Protection**:
- HSM-backed key derivation (Option A, Option B, DUKPT)
- PQC-wrapped issuer master keys
- Cryptogram verification chain

---

### Use Case 3: Real-Time Payment Protection (FedNow)

**Threat**: Real-time payments require sub-50ms security with quantum resistance.

**Solution**: Optimized PQC profile for low-latency operations.

```python
from ai_engine.domains.banking.core.domain_profile import (
    BankingPQCProfile,
    BankingSubdomain,
    BankingSecurityConstraints,
)
from ai_engine.domains.banking.protocols.payments.domestic.fednow import (
    FedNowBuilder,
)

# Load real-time payment profile (optimized for <50ms)
profile = BankingPQCProfile.for_subdomain(
    subdomain=BankingSubdomain.REAL_TIME_PAYMENTS,
    constraints=BankingSecurityConstraints(
        max_latency_ms=50,
        min_throughput_tps=10_000,
        pqc_mandatory=True,
    ),
)
# Auto-selects: ML-KEM-512 + ML-DSA-44 (fastest PQC algorithms)

# Build instant payment with quantum-safe security
builder = FedNowBuilder(profile=profile)
payment = builder.create_instant_credit(
    debtor_account="123456789",
    creditor_account="987654321",
    amount=500.00,
    currency="USD",
)

# Process with guaranteed latency
response = await fednow_gateway.submit(
    payment,
    timeout_ms=50,
    security_profile=profile,
)
```

**Protection**:
- Sub-50ms quantum-safe operations
- Optimized algorithm selection
- HSM acceleration for PQC

---

### Use Case 4: SWIFT Message Security

**Threat**: Stored SWIFT messages could be decrypted by future quantum computers ("harvest now, decrypt later").

**Solution**: PQC encryption for message storage and hybrid signatures.

```python
from ai_engine.domains.banking.protocols.messaging.swift import (
    SwiftParser,
    MT103Message,
    MT103Builder,
)
from ai_engine.domains.banking.security.pqc import HybridEncryption

# Parse incoming SWIFT message
parser = SwiftParser()
message = parser.parse(raw_swift_mt103)
mt103 = MT103Message.from_swift_message(message)

# Re-encrypt for quantum-safe storage
hybrid = HybridEncryption(
    classical="AES-256-GCM",
    pqc="ML-KEM-768",
)

encrypted_archive = hybrid.encrypt(
    plaintext=message.raw_content,
    aad=f"MT103:{mt103.transaction_reference}".encode(),
)

# Store with quantum-safe protection
await archive.store(
    reference=mt103.transaction_reference,
    encrypted_data=encrypted_archive,
    retention_years=7,  # Regulatory requirement
)
```

**Protection**:
- Hybrid encryption (AES + ML-KEM)
- 20+ year protection against quantum attacks
- Compliant with 7-year retention requirements

---

### Use Case 5: HSM Key Migration

**Threat**: Existing HSM keys vulnerable to future quantum attacks.

**Solution**: Multi-cloud HSM with PQC key wrapping.

```python
from ai_engine.automation import KeyLifecycleManager, KeyType
from ai_engine.domains.banking.security.hsm import HSMPool
from ai_engine.domains.banking.security.hsm.cloud import (
    AWSCloudHSM,
    AzureManagedHSM,
)

# Configure multi-cloud HSM pool
hsm_pool = HSMPool([
    AWSCloudHSM(config={"cluster_id": "cluster-1", "pqc_enabled": True}),
    AzureManagedHSM(config={"vault_url": "https://bank.managedhsm.azure.net"}),
])

# Initialize key lifecycle manager
key_manager = KeyLifecycleManager(hsm_pool=hsm_pool)

# Migrate keys with PQC protection
for key in await legacy_hsm.list_keys():
    # Wrap with PQC before migration
    pqc_wrapped = await hsm_pool.wrap_key_pqc(
        key_material=key,
        algorithm="ML-KEM-1024",  # Highest security for master keys
    )

    # Import to cloud HSM
    new_key = await key_manager.import_key(
        wrapped_key=pqc_wrapped,
        key_type=key.type,
        tags={"migrated_from": "legacy", "pqc_wrapped": "true"},
    )

    # Set rotation policy
    await key_manager.set_rotation_policy(
        key_id=new_key.key_id,
        rotation_period_days=90,
        auto_rotate=True,
    )
```

**Protection**:
- PQC-wrapped key migration
- Multi-cloud redundancy
- Automated rotation policies

---

### Use Case 6: Trading System Security (FIX Protocol)

**Threat**: High-frequency trading requires microsecond latency with quantum resistance.

**Solution**: Software-based PQC for lowest latency.

```python
from ai_engine.domains.banking.protocols.trading.fix import (
    FixParser,
    FixBuilder,
    FixMessage,
)
from ai_engine.domains.banking.core.domain_profile import (
    BankingPQCProfile,
    BankingSubdomain,
)

# Load trading profile (1ms latency requirement)
profile = BankingPQCProfile.for_subdomain(BankingSubdomain.TRADING)
# Uses software crypto for lowest latency (no HSM round-trip)

# Parse order with quantum-safe verification
parser = FixParser(version="FIX.4.4", security_profile=profile)
order = parser.parse_and_verify(raw_fix_message)

# Execute and sign response
execution = execute_order(order)

builder = FixBuilder(version="FIX.4.4")
exec_report = builder.create_execution_report(
    order_id=order.get_field(11),
    exec_id=execution.exec_id,
    exec_type="F",
    symbol=order.get_field(55),
)

# Sign with fast PQC signature
signature = profile.sign(exec_report.to_bytes())
```

**Protection**:
- Sub-millisecond PQC operations
- Software-based for lowest latency
- Quantum-safe order authentication

---

### Use Case 7: 3D Secure Authentication

**Threat**: E-commerce authentication tokens could be forged by quantum computers.

**Solution**: PQC-enhanced CAVV verification.

```python
from ai_engine.domains.banking.security.crypto.verification import (
    ThreeDSVerifier,
    ThreeDSVerificationData,
)

# Initialize 3DS verifier with PQC
verifier = ThreeDSVerifier(pqc_enabled=True)

# Verify CAVV with quantum-safe MAC
data = ThreeDSVerificationData(
    cavv=bytes.fromhex("AAABBCCCDDDEEE..."),
    eci="05",
    ds_transaction_id="f25084f0-5b16-4c0a-ae5d-b24808a95e4b",
    message_version="2.2.0",
    transaction_status="Y",
)

result = verifier.verify_cavv(data, acquirer_key)

if result.status == VerificationStatus.VALID:
    # ECI 05 = full authentication, liability shift
    process_with_liability_shift()
```

---

### Use Case 8: Compliance Monitoring

**Threat**: Regulatory requirements for quantum readiness (NIST, DORA).

**Solution**: Continuous compliance monitoring with PQC readiness assessment.

```python
from ai_engine.domains.banking.core.compliance_context import (
    ComplianceContext,
    ComplianceFramework,
)
from ai_engine.observability import AuditLogger

# Initialize compliance context
context = ComplianceContext()
context.add_framework(ComplianceFramework.NIST_PQC)
context.add_framework(ComplianceFramework.PCI_DSS_4_0)
context.add_framework(ComplianceFramework.DORA)

# Assess PQC readiness
pqc_assessment = await context.assess_pqc_readiness()
print(f"PQC Migration Progress: {pqc_assessment.percent_complete}%")
print(f"Systems using PQC: {pqc_assessment.pqc_enabled_systems}")
print(f"Systems pending: {pqc_assessment.pending_systems}")

# Generate compliance report
report = context.generate_compliance_report(
    frameworks=[ComplianceFramework.NIST_PQC],
    include_evidence=True,
)

# Tamper-evident audit logging
audit_logger = AuditLogger()
audit_logger.log_compliance_event(
    framework="NIST_PQC",
    control_id="PQC-MIGRATION-001",
    status="IN_PROGRESS",
    evidence=pqc_assessment.to_dict(),
)
```

---

## Protocol Support

### Payment Protocols

| Protocol | Type | PQC Support | Use Case |
|----------|------|-------------|----------|
| **ACH/NACHA** | Domestic | ✅ | Payroll, bill pay |
| **FedWire** | Domestic | ✅ | Wire transfers |
| **FedNow** | Domestic | ✅ | Real-time payments |
| **SWIFT MT/MX** | International | ✅ | Correspondent banking |
| **ISO 20022** | Universal | ✅ | Structured payments |
| **ISO 8583** | Card | ✅ | ATM/POS |
| **EMV** | Card | ✅ | Chip cards |
| **3D Secure** | Card | ✅ | E-commerce |

### Trading Protocols

| Protocol | Type | PQC Support | Use Case |
|----------|------|-------------|----------|
| **FIX 4.2/4.4/5.0** | Orders | ✅ | Trading |
| **FpML** | Derivatives | ✅ | OTC derivatives |

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          QBITEL Platform                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                   Zero-Touch Automation Engine                       │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │ │
│  │  │   Key     │ │Certificate│ │   Self-   │ │   Auto-   │           │ │
│  │  │ Lifecycle │ │Automation │ │  Healing  │ │  Config   │           │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│                        ┌───────────┴───────────┐                          │
│                        ▼                       ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                AI-Powered Protocol Discovery                         │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │ │
│  │  │  Traffic  │ │  Pattern  │ │  Grammar  │ │  Parser   │           │ │
│  │  │ Analysis  │ │  Learning │ │ Inference │ │ Generator │           │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                Post-Quantum Cryptography Layer                       │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │ │
│  │  │  ML-KEM   │ │  ML-DSA   │ │  Hybrid   │ │    Key    │           │ │
│  │  │(FIPS 203) │ │(FIPS 204) │ │  Crypto   │ │  Wrapping │           │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                   Cloud-Native Security                              │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │ │
│  │  │ Container │ │  Service  │ │  Runtime  │ │  Network  │           │ │
│  │  │ Security  │ │   Mesh    │ │Protection │ │ Policies  │           │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                  Multi-Cloud HSM Abstraction                         │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │ │
│  │  │    AWS    │ │   Azure   │ │    GCP    │ │ On-Premise│           │ │
│  │  │ CloudHSM  │ │    HSM    │ │ Cloud HSM │ │  Thales   │           │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                   Policy Engine & Compliance                         │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │ │
│  │  │  Banking  │ │Compliance │ │   Audit   │ │   Alert   │           │ │
│  │  │ Policies  │ │ Frameworks│ │  Logging  │ │  Manager  │           │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Zero-Touch Automation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Zero-Touch Automation Flow                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │
│  │   Trigger   │─────▶│   Policy    │─────▶│  Automated  │              │
│  │   Events    │      │  Evaluation │      │   Action    │              │
│  └─────────────┘      └─────────────┘      └─────────────┘              │
│        │                    │                    │                       │
│        ▼                    ▼                    ▼                       │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │
│  │• Key Expiry │      │• Pre-defined│      │• Key Rotate │              │
│  │• Cert Expiry│      │  Banking    │      │• Cert Renew │              │
│  │• Health Fail│      │  Policies   │      │• HSM Failovr│              │
│  │• Compliance │      │• Compliance │      │• Config Updt│              │
│  │  Drift      │      │  Rules      │      │• Adapter Gen│              │
│  └─────────────┘      └─────────────┘      └─────────────┘              │
│                                                   │                       │
│                                                   ▼                       │
│                              ┌─────────────────────────────────┐         │
│                              │         Audit Trail             │         │
│                              │  • Complete action logging      │         │
│                              │  • Tamper-evident records       │         │
│                              │  • Compliance evidence          │         │
│                              └─────────────────────────────────┘         │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Compliance Frameworks

| Framework | Scope | PQC Requirements |
|-----------|-------|------------------|
| **NIST PQC** | Federal systems | ML-KEM, ML-DSA mandatory by 2030 |
| **PCI-DSS 4.0** | Payment cards | Encryption, key management |
| **DORA** | EU financial | Operational resilience |
| **Basel III/IV** | Risk management | Data protection |
| **GDPR** | Data privacy | Encryption at rest |

---

## Getting Started

### Installation

```bash
pip install qbitel[banking,pqc]
```

### Quick Start

```python
from ai_engine.domains.banking.core.domain_profile import (
    BankingPQCProfile,
    BankingSubdomain,
)
from ai_engine.automation import AutomationEngine

# 1. Select your banking subdomain
profile = BankingPQCProfile.for_subdomain(BankingSubdomain.WIRE_TRANSFERS)

# 2. Initialize automation engine with PQC
engine = AutomationEngine(profile=profile, pqc_enabled=True)
engine.start()

# 3. Your banking system is now quantum-safe!
```

---

## API Reference

### Core Security Classes

| Class | Module | Description |
|-------|--------|-------------|
| `BankingPQCProfile` | `core.domain_profile` | PQC algorithm profiles |
| `HSMProvider` | `security.hsm` | HSM abstraction |
| `BankingSecurityPolicy` | `core.security_policy` | Pre-defined banking policies |

### Zero-Touch Automation Classes

| Class | Module | Description |
|-------|--------|-------------|
| `AutomationEngine` | `automation` | Main orchestration engine |
| `KeyLifecycleManager` | `automation` | Automated key management |
| `CertificateAutomation` | `automation` | Automated certificate lifecycle |
| `SelfHealingOrchestrator` | `automation` | Automated failure recovery |
| `ConfigurationManager` | `automation` | Auto-configuration |
| `ProtocolAdapterGenerator` | `automation` | Auto-generate protocol adapters |
| `PolicyEngine` | `policy` | Policy-driven automation |

### PQC Classes

| Class | Module | Description |
|-------|--------|-------------|
| `MLKEM768` | `security.pqc` | Key encapsulation |
| `MLDSA65` | `security.pqc` | Digital signatures |
| `HybridKeyExchange` | `security.pqc` | Hybrid crypto |

### Protocol Classes

| Class | Module | Description |
|-------|--------|-------------|
| `ProtocolDiscoveryOrchestrator` | `discovery` | AI protocol discovery |
| `SwiftParser` | `protocols.messaging.swift` | SWIFT parsing |
| `EMVCryptogramVerifier` | `security.crypto.verification` | EMV verification |

---

*QBITEL - Enterprise-Grade Quantum-Safe Security Platform*

**Protecting legacy systems and modern infrastructure against quantum computing threats.**
