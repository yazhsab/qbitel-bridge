# Domain-Specific Post-Quantum Cryptography Implementation Plan

## QBITEL / QBITEL Bridge - Strategic Roadmap

**Version**: 1.0
**Date**: December 2025
**Status**: Strategic Planning Phase

---

## Executive Summary

Based on comprehensive analysis of the PQC research landscape and the existing QBITEL codebase, this document outlines a strategic implementation plan to extend the platform's quantum-safe capabilities into domain-specific constrained environments.

### Current State Assessment

QBITEL already possesses **production-grade PQC infrastructure**:
- ✅ NIST Level 5 algorithms (Kyber-1024, Dilithium-5)
- ✅ Hybrid encryption (Classical + PQC)
- ✅ HSM integration via PKCS#11
- ✅ Automated key lifecycle management (8,461 LOC Rust implementation)
- ✅ Service mesh integration (Istio/Envoy)
- ✅ Container signing with Dilithium
- ✅ Air-gapped deployment capability

### Strategic Opportunity

The research identifies critical gaps between NIST standards and enterprise reality. QBITEL is uniquely positioned to address these gaps through domain-specific optimizations targeting:

1. **Healthcare** - Constrained medical devices (64KB RAM, 10+ year battery)
2. **Automotive V2X** - Sub-10ms latency, 1000+ msg/sec verification
3. **Aviation** - Bandwidth-constrained channels (600bps-2.4kbps)
4. **Industrial OT/ICS** - Safety-critical certification requirements

---

## Part 1: Gap Analysis - Research vs. Current Capabilities

### 1.1 Signature Size Challenge

| Metric | Current Implementation | Research Target | Gap |
|--------|----------------------|-----------------|-----|
| ML-DSA Signature | 4,595 bytes | <1KB for constrained | 4.5x reduction needed |
| Certificate Chain | ~15-17KB | <10KB (ossification) | Compression required |
| TLS Handshake | ~17KB PQC overhead | <5KB for mobile | Hybrid optimization |

**Current Code Location**: `/rust/dataplane/crates/pqc_tls/src/kyber.rs`

### 1.2 Performance Benchmarks vs. Requirements

| Domain | Required Latency | Current Capability | Status |
|--------|-----------------|-------------------|--------|
| Enterprise TLS | <50ms | <10ms | ✅ Exceeds |
| V2X Authentication | <10ms | Not optimized | ⚠️ Gap |
| Medical Device | Battery-constrained | Full power | ⚠️ Gap |
| Aviation ACARS | 600bps bandwidth | Standard PQC | ❌ Critical Gap |

### 1.3 Algorithm Coverage

| Algorithm | Status | Use Case |
|-----------|--------|----------|
| Kyber-1024 | ✅ Implemented | Key encapsulation |
| Dilithium-5 | ✅ Implemented | Digital signatures |
| ML-KEM-768 | ⚠️ Not implemented | Hybrid TLS |
| SLH-DSA/SPHINCS+ | ⚠️ Partial (dependency only) | Backup signatures |
| LMS/XMSS | ❌ Not implemented | Stateful hash-based |
| Falcon | ❌ Not implemented | Compact signatures |

---

## Part 2: Implementation Phases

### Phase 1: Foundation Enhancement (Q1 2026)

**Objective**: Strengthen core PQC infrastructure and add missing algorithms

#### 1.1 Algorithm Expansion

```
Priority 1: ML-KEM-768 (Hybrid TLS standard)
Priority 2: Falcon-512/1024 (50% smaller signatures than Dilithium)
Priority 3: SLH-DSA production integration
Priority 4: LMS/XMSS for stateful applications
```

**Implementation Tasks**:

| Task | Location | Effort | Priority |
|------|----------|--------|----------|
| Add ML-KEM-768 to Rust crate | `/rust/dataplane/crates/pqc_tls/` | Medium | P1 |
| Integrate Falcon via pqcrypto-falcon | `/rust/dataplane/crates/pqc_tls/` | Medium | P1 |
| SLH-DSA wrapper implementation | `/ai_engine/cloud_native/` | Low | P2 |
| LMS/XMSS state management | New module | High | P2 |
| Update certificate manager | `qkd_certificate_manager.py` | Medium | P1 |

**Files to Modify**:
- `/rust/dataplane/crates/pqc_tls/Cargo.toml` - Add new dependencies
- `/rust/dataplane/crates/pqc_tls/src/lib.rs` - New algorithm exports
- `/ai_engine/requirements.txt` - Python dependencies

#### 1.2 Hybrid Certificate Implementation

Based on IETF draft standards (expected early 2026):

```rust
// Proposed structure in /rust/dataplane/crates/pqc_tls/src/hybrid.rs
pub struct HybridCertificate {
    classical: X509Certificate,      // ECDSA-P384
    pqc: PQCCertificate,             // ML-DSA or Falcon
    binding: CertificateBinding,      // Cryptographic binding
}
```

**Key Considerations**:
- Support both composite and non-composite formats
- Backward compatibility with classical-only clients
- Size optimization through compression

#### 1.3 TLS 1.3 PQC Integration

Extend existing Rust TLS implementation:

```rust
// /rust/dataplane/crates/pqc_tls/src/tls13.rs
pub enum HybridKexGroup {
    X25519MLKEM768,      // Chrome/Cloudflare default
    P384MLKEM1024,       // Enterprise preference
    X25519Kyber768,      // Legacy compatibility
}
```

**Performance Target**: <0.5% latency increase on initial handshake

---

### Phase 2: Domain-Specific Modules (Q2-Q3 2026)

#### 2.1 Healthcare Module

**Target Constraints**:
- 64KB RAM (pacemaker-class devices)
- 10+ year battery life
- FDA 510(k) / PMA compliance

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Healthcare PQC Module                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Lightweight    │  │ External       │  │ FHIR       │ │
│  │ PQC (ML-KEM-512)│ │ Security Shield│  │ PQC Profile│ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Battery-Aware  │  │ HL7 v2/FHIR   │  │ HIPAA      │ │
│  │ Key Management │  │ Integration    │  │ Compliance │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | Description | Effort |
|------|-------------|--------|
| ML-KEM-512 optimization | Reduced security level for constrained devices | High |
| External security shield | Proxy-based PQC for legacy implantables | High |
| FHIR PQC security profile | Quantum-safe SMART-on-FHIR | Medium |
| Battery-aware scheduling | Defer crypto operations to optimal times | Medium |
| HL7 integration | PQC transport for HL7 v2 messages | Low |

**New Files**:
```
/ai_engine/domains/healthcare/
├── __init__.py
├── lightweight_pqc.py          # Optimized algorithms
├── security_shield.py          # External protection proxy
├── fhir_pqc_profile.py         # FHIR security extensions
├── hl7_transport.py            # HL7 v2 integration
├── battery_scheduler.py        # Power-aware operations
└── fda_compliance.py           # Regulatory requirements
```

**Key Innovation**: External Security Shield

Concept from MIT's IMDShield research - provide PQC protection for legacy devices that cannot be modified:

```python
# /ai_engine/domains/healthcare/security_shield.py
class MedicalDeviceSecurityShield:
    """
    External proxy providing PQC protection for legacy medical devices.
    Operates between device and network, handling all quantum-safe crypto.
    """

    def __init__(self, device_profile: DeviceProfile):
        self.pqc_engine = LightweightPQCEngine(memory_limit_kb=32)
        self.session_cache = SecureSessionCache()

    async def protect_outbound(self, plaintext: bytes) -> bytes:
        """Encrypt device traffic with PQC before transmission"""

    async def protect_inbound(self, ciphertext: bytes) -> bytes:
        """Decrypt incoming PQC traffic for legacy device"""
```

#### 2.2 Automotive V2X Module

**Target Constraints**:
- 100ms BSM broadcast interval (10 Hz)
- <10ms signature verification
- 1,000+ messages/second in dense scenarios
- 15-20 year fleet turnover timeline

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Automotive V2X PQC Module                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Batch          │  │ Implicit Cert  │  │ SCMS       │ │
│  │ Verification   │  │ PQC Scheme     │  │ Integration│ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ ARMv8          │  │ Legacy Vehicle │  │ SAE J2735  │ │
│  │ Optimization   │  │ Compatibility  │  │ Compliance │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | Description | Effort |
|------|-------------|--------|
| Batch verification engine | 65-90% signature operation reduction | High |
| PQ implicit certificates | Research PQCMC integration | High |
| ARMv8 NEON optimization | Target NXP MK6 chipset | High |
| P2PCD adaptation | Certificate distribution for large certs | Medium |
| IEEE 1609.2 extension | PQC SPDU format support | Medium |
| Backward compatibility | Hybrid mode for legacy vehicles | Medium |

**New Files**:
```
/ai_engine/domains/automotive/
├── __init__.py
├── batch_verification.py       # Parallel signature verification
├── implicit_certificates.py    # PQC implicit cert scheme
├── armv8_optimization.py       # Platform-specific optimization
├── v2x_protocol.py             # IEEE 1609.2 / SAE J2735
├── p2pcd_adapter.py            # Certificate distribution
└── fleet_compatibility.py      # Legacy vehicle support
```

**Key Innovation**: Batch Verification

Based on NDSS 2024 research, achieve 65-90% reduction in signature operations:

```rust
// /rust/dataplane/crates/pqc_tls/src/batch.rs
pub struct BatchVerifier {
    pending: Vec<SignedMessage>,
    batch_size: usize,           // Optimal: 32-64 messages
    parallel_workers: usize,     // Match CPU cores
}

impl BatchVerifier {
    pub async fn verify_batch(&mut self) -> Vec<VerificationResult> {
        // Aggregate signature verification
        // Use SIMD for parallel hash computation
        // Amortize lattice operations across batch
    }
}
```

**Performance Target**: 1,000+ verifications/second with <10ms P99 latency

#### 2.3 Aviation Module

**Target Constraints**:
- VHF ACARS: 2.4 kbps
- Classic SATCOM: 600 bps
- 30+ year aircraft lifecycle
- DO-326A/ED-202A certification

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Aviation PQC Module                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Signature      │  │ LDACS         │  │ ADS-B      │ │
│  │ Compression    │  │ Integration    │  │ Auth       │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ ARINC 653     │  │ Bandwidth      │  │ DO-326A    │ │
│  │ Integration    │  │ Optimization   │  │ Compliance │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | Description | Effort |
|------|-------------|--------|
| Signature compression | Target <1KB for VHF ACARS | Critical |
| LDACS PQC profile | Post-SIKE algorithm integration | High |
| ADS-B authenticator | Lightweight position/speed auth | High |
| ARINC 653 partition | IMA-compliant PQC module | Medium |
| Delta encoding | Send signature differences | Medium |
| Session resumption | Minimize key exchange frequency | Low |

**New Files**:
```
/ai_engine/domains/aviation/
├── __init__.py
├── signature_compression.py    # Advanced compression
├── ldacs_profile.py            # L-band integration
├── adsb_authenticator.py       # ADS-B security
├── arinc653_partition.py       # IMA integration
├── bandwidth_optimizer.py      # Channel-aware crypto
└── certification_support.py    # DO-326A evidence
```

**Key Innovation**: Adaptive Signature Compression

For 600bps SATCOM channels, standard PQC is impossible. Novel approach:

```python
# /ai_engine/domains/aviation/signature_compression.py
class AdaptiveSignatureCompressor:
    """
    Multi-strategy compression for bandwidth-constrained aviation channels.
    Achieves 60-80% size reduction through:
    1. Zstandard dictionary compression with domain-specific training
    2. Delta encoding for sequential messages
    3. Session-based signature aggregation
    4. Selective field authentication (position vs. full message)
    """

    def __init__(self, channel_bandwidth_bps: int):
        self.strategy = self._select_strategy(channel_bandwidth_bps)

    def compress_signature(
        self,
        signature: bytes,
        previous_signatures: List[bytes]
    ) -> CompressedSignature:
        """
        Target: <600 bytes for VHF ACARS (2.4kbps)
        Target: <150 bytes for SATCOM (600bps) using aggregation
        """
```

#### 2.4 Industrial OT/ICS Module

**Target Constraints**:
- IEC 61508 SIL 1-4 certification
- Deterministic timing (WCET analysis)
- 10+ year operational lifecycle
- Air-gapped networks

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Industrial OT/ICS PQC Module                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ Deterministic  │  │ Safety        │  │ Protocol   │ │
│  │ PQC Engine     │  │ Certification │  │ Adaptation │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ SCADA/Modbus  │  │ DNP3          │  │ IEC 62443  │ │
│  │ Integration    │  │ Secure Auth   │  │ Compliance │ │
│  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation Tasks**:

| Task | Description | Effort |
|------|-------------|--------|
| Constant-time implementation | Eliminate timing side channels | Critical |
| WCET analysis framework | Safety certification support | High |
| Modbus/TCP PQC wrapper | Protocol-level protection | Medium |
| DNP3 Secure Auth v6 | PQC extension | Medium |
| IEC 62443 compliance | Security level mapping | Medium |
| Offline key provisioning | Air-gapped key management | Low |

**New Files**:
```
/ai_engine/domains/industrial/
├── __init__.py
├── deterministic_pqc.py        # Constant-time implementation
├── wcet_analyzer.py            # Timing analysis
├── modbus_wrapper.py           # Modbus/TCP protection
├── dnp3_secure_auth.py         # DNP3 SA v6 extension
├── iec62443_compliance.py      # Security level mapping
└── offline_provisioning.py     # Air-gapped key management
```

**Key Innovation**: Deterministic PQC for Safety-Critical Systems

```rust
// /rust/dataplane/crates/pqc_tls/src/deterministic.rs
/// Constant-time PQC implementation for IEC 61508 / ISO 26262
/// All operations complete within guaranteed WCET bounds
pub struct DeterministicPQCEngine {
    wcet_sign_us: u64,      // Worst-case signing time
    wcet_verify_us: u64,    // Worst-case verification time
    wcet_encap_us: u64,     // Worst-case encapsulation time
    jitter_budget_us: u64,  // Allowed timing variance
}

impl DeterministicPQCEngine {
    /// Returns Some(signature) only if completed within WCET
    /// Returns None if operation would exceed safety bounds
    pub fn sign_with_deadline(
        &self,
        message: &[u8],
        deadline_us: u64
    ) -> Option<Signature> {
        // Pre-check: Can we meet the deadline?
        if deadline_us < self.wcet_sign_us {
            return None;
        }
        // Execute with timing guards
    }
}
```

---

### Phase 3: Advanced Cryptographic Primitives (Q4 2026)

#### 3.1 Post-Quantum Zero-Knowledge Proofs

**Current State**: 450× size gap vs. classical (16KB-58KB vs. 128 bytes)

**Implementation Approach**:

| Primitive | Target Size | Use Case |
|-----------|-------------|----------|
| LaBRADOR (R1CS) | 58KB | General ZK proofs |
| ISW21 (designated) | 16KB | Identity verification |
| LaZer integration | Variable | Anonymous credentials |

**New Files**:
```
/ai_engine/crypto/zkp/
├── __init__.py
├── labrador.py                 # LaBRADOR wrapper
├── designated_verifier.py      # ISW21 implementation
├── anonymous_credentials.py    # PQ credential scheme
└── kyber_key_proofs.py         # Prove Kyber key knowledge
```

#### 3.2 Post-Quantum Anonymous Credentials

Target: <100KB total credential size (vs. current 317-724KB)

**Strategy**: Adapt Cloudflare's Phoenix signature research

```python
# /ai_engine/crypto/zkp/anonymous_credentials.py
class PQAnonymousCredential:
    """
    Post-quantum anonymous credential based on lattice assumptions.

    Target sizes:
    - Credential: ~80KB (vs. 317-724KB state-of-art)
    - Presentation proof: <7KB
    - Verification: <50ms
    """

    def issue(self, attributes: Dict[str, Any]) -> Credential:
        """Issue credential with hidden attributes"""

    def present(
        self,
        credential: Credential,
        disclosed: Set[str],
        predicates: List[Predicate]
    ) -> Presentation:
        """Generate ZK presentation revealing only selected attributes"""
```

#### 3.3 Post-Quantum Verifiable Random Functions

Based on LB-VRF research (Esgin et al.):

```python
# /ai_engine/crypto/vrf/
class LatticePQVRF:
    """
    Lattice-based VRF for quantum-resistant randomness.

    Sizes:
    - VRF value: 84 bytes
    - Proof: ~5KB

    Performance:
    - Evaluation: ~3ms
    - Verification: ~1ms
    """
```

---

### Phase 4: Ecosystem Integration (Q1-Q2 2027)

#### 4.1 Cloud HSM Integration

Extend existing PKCS#11 integration to cloud HSMs:

| Provider | Service | Status |
|----------|---------|--------|
| AWS | CloudHSM | Planned |
| Azure | Managed HSM | Planned |
| GCP | Cloud HSM | Planned |
| Thales | Luna 7.9.0 | Available now |
| Utimaco | Quantum Protect | Available now |

**Files to Extend**:
- `/rust/dataplane/crates/pqc_tls/src/hsm.rs`

#### 4.2 5G/6G Network Integration

Based on 3GPP SA3 PQC roadmap (Release 19/20):

| Component | Integration Point | Priority |
|-----------|------------------|----------|
| SUCI protection | KEMSUCI scheme | High |
| Base station auth | Hybrid certificates | High |
| Edge computing | PQES offloading | Medium |
| IoT authentication | Lightweight KEM | Medium |

**New Files**:
```
/ai_engine/domains/telecom/
├── __init__.py
├── kemsuci.py                  # Post-quantum SUCI
├── base_station_auth.py        # 5G infrastructure
├── edge_offload.py             # PQES integration
└── iot_lightweight.py          # Constrained IoT devices
```

#### 4.3 Blockchain/DLT Integration

Leverage existing lattice-based HE (inherently PQ-safe):

| Platform | Integration | Use Case |
|----------|-------------|----------|
| Algorand | LB-VRF | Proof-of-stake |
| Hyperledger | PQ signatures | Permissioned chains |
| Ethereum L2 | zkSTARKs | Scalability |

---

## Part 3: Technical Implementation Details

### 3.1 Library Strategy

**Rust (Performance-Critical)**:
```toml
# /rust/dataplane/crates/pqc_tls/Cargo.toml additions
[dependencies]
pqcrypto-mlkem = "0.1"          # ML-KEM (new NIST name)
pqcrypto-falcon = "0.3"          # Falcon signatures
pqcrypto-sphincsplus = "0.7"     # SPHINCS+/SLH-DSA
lms = "0.1"                      # LMS signatures
xmss = "0.1"                     # XMSS signatures
```

**Python (Flexibility)**:
```python
# /ai_engine/requirements.txt additions
falcon-py>=0.1.0                 # Falcon signatures
sphincs-py>=0.1.0                # SPHINCS+
liboqs-python>=0.9.0             # Updated OQS
```

### 3.2 API Design

**Unified PQC Interface**:

```python
# /ai_engine/crypto/pqc_unified.py
from enum import Enum
from typing import Union

class PQCAlgorithm(Enum):
    # Key Encapsulation
    KYBER_512 = "kyber-512"
    KYBER_768 = "kyber-768"
    KYBER_1024 = "kyber-1024"
    MLKEM_512 = "ml-kem-512"
    MLKEM_768 = "ml-kem-768"
    MLKEM_1024 = "ml-kem-1024"

    # Signatures
    DILITHIUM_2 = "dilithium-2"
    DILITHIUM_3 = "dilithium-3"
    DILITHIUM_5 = "dilithium-5"
    FALCON_512 = "falcon-512"
    FALCON_1024 = "falcon-1024"
    SPHINCS_SHA2_128F = "sphincs-sha2-128f"
    SPHINCS_SHA2_256F = "sphincs-sha2-256f"
    LMS = "lms"
    XMSS = "xmss"

class DomainProfile(Enum):
    ENTERPRISE = "enterprise"           # Standard security
    HEALTHCARE = "healthcare"           # Constrained devices
    AUTOMOTIVE = "automotive"           # Real-time V2X
    AVIATION = "aviation"               # Bandwidth constrained
    INDUSTRIAL = "industrial"           # Safety-critical
    TELECOM = "telecom"                 # 5G/6G networks

class PQCEngine:
    """
    Unified PQC engine with domain-specific optimization.
    """

    def __init__(
        self,
        domain: DomainProfile,
        algorithm: PQCAlgorithm = None,    # Auto-select if None
        hybrid: bool = True                 # Classical + PQC
    ):
        self.domain = domain
        self.algorithm = algorithm or self._auto_select_algorithm()
        self.hybrid = hybrid

    def _auto_select_algorithm(self) -> PQCAlgorithm:
        """Select optimal algorithm for domain constraints"""
        domain_defaults = {
            DomainProfile.ENTERPRISE: PQCAlgorithm.KYBER_1024,
            DomainProfile.HEALTHCARE: PQCAlgorithm.MLKEM_512,
            DomainProfile.AUTOMOTIVE: PQCAlgorithm.FALCON_512,
            DomainProfile.AVIATION: PQCAlgorithm.FALCON_512,
            DomainProfile.INDUSTRIAL: PQCAlgorithm.KYBER_768,
            DomainProfile.TELECOM: PQCAlgorithm.MLKEM_768,
        }
        return domain_defaults[self.domain]
```

### 3.3 Performance Optimization Strategy

**Hardware Acceleration Targets**:

| Platform | Optimization | Expected Speedup |
|----------|-------------|------------------|
| x86-64 AVX-512 | SIMD lattice ops | 3-5× |
| x86-64 AVX2 | SIMD fallback | 2-3× |
| ARMv8 NEON | Mobile/automotive | 2-4× |
| ARMv8 SVE2 | Server ARM | 3-5× |
| RISC-V V | Emerging | 2-3× |

**Existing Optimization** (from `/rust/dataplane/crates/pqc_tls/src/config.rs`):
```rust
#[cfg(target_feature = "avx512f")]
mod avx512_optimized;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
mod avx2_optimized;

#[cfg(target_arch = "aarch64")]
mod neon_optimized;
```

### 3.4 Testing Strategy

**Test Categories**:

| Category | Coverage Target | Location |
|----------|-----------------|----------|
| Unit Tests | 90%+ | `/ai_engine/tests/crypto/` |
| Integration | 80%+ | `/ai_engine/tests/integration/` |
| Performance | All algorithms | `/tests/perf/` |
| Conformance | NIST vectors | `/tests/conformance/` |
| Fuzz Testing | All inputs | `/tests/fuzz/` |
| Domain-specific | Each module | `/ai_engine/domains/*/tests/` |

---

## Part 4: Compliance & Certification Roadmap

### 4.1 Regulatory Timeline

| Regulation | Deadline | Status |
|------------|----------|--------|
| NSA CNSA 2.0 (prefer PQC) | 2025 | ✅ Ready |
| NSA CNSA 2.0 (exclusive) | 2030 | 🔄 On track |
| OMB M-23-02 | 2035 | 🔄 Planning |
| UK NCSC Phase 1 | 2028 | 🔄 On track |
| UK NCSC Phase 2 | 2031 | 🔄 Planning |
| UK NCSC Phase 3 | 2035 | 🔄 Planning |

### 4.2 Certification Targets

| Certification | Target Level | Timeline |
|---------------|-------------|----------|
| FIPS 140-3 | Level 2+ | Q2 2026 |
| Common Criteria | EAL 4+ | Q4 2026 |
| ISO 27001 | Certified | Q1 2026 |
| SOC 2 Type II | Compliant | Q2 2026 |
| FedRAMP | High | Q4 2026 |
| HIPAA | Compliant | Q1 2026 |
| PCI-DSS | Level 1 | Q2 2026 |

### 4.3 Domain-Specific Certifications

| Domain | Certification | Target |
|--------|--------------|--------|
| Healthcare | FDA 510(k)/PMA cyber | Q3 2026 |
| Automotive | ISO 26262 ASIL B | Q4 2026 |
| Aviation | DO-326A/ED-202A | 2027 |
| Industrial | IEC 62443 SL2+ | Q2 2026 |

---

## Part 5: Resource Requirements

### 5.1 Development Team

| Role | Count | Focus Area |
|------|-------|------------|
| Cryptography Engineers | 2-3 | Algorithm implementation |
| Rust Developers | 2 | Performance optimization |
| Python Developers | 2 | Domain modules |
| Domain Specialists | 4 | Healthcare/Auto/Aviation/Industrial |
| Security Researchers | 1-2 | Novel primitives |
| QA Engineers | 2 | Testing & validation |
| Compliance Specialist | 1 | Certification |

### 5.2 Infrastructure

| Resource | Specification | Purpose |
|----------|--------------|---------|
| Dev Servers | 32-core, 128GB RAM | Build & test |
| HSM (Dev) | Thales Luna Network | Key management testing |
| CI/CD | GitHub Actions + Self-hosted | Continuous testing |
| Test Vehicles | 2-3 V2X equipped | Automotive testing |
| Medical Device Lab | FDA-compliant | Healthcare testing |

### 5.3 Partnerships

| Partner Type | Purpose | Priority |
|--------------|---------|----------|
| Automotive OEM | V2X validation | High |
| Medical Device Mfr | FDA pathway | High |
| Aviation Integrator | DO-326A evidence | Medium |
| HSM Vendor | Hardware integration | High |
| Academic | Novel research | Medium |

---

## Part 6: Risk Assessment

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Algorithm weakness discovered | Low | Critical | Crypto-agility architecture |
| Performance targets missed | Medium | High | Early benchmarking, fallback options |
| Certification delays | Medium | High | Early engagement, parallel tracks |
| Interoperability issues | Medium | Medium | Standards participation, testing |

### 6.2 Market Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Standards changes | Medium | Medium | Active IETF/NIST participation |
| Competitor advancement | Medium | Medium | Patent portfolio, innovation speed |
| Customer adoption delays | Medium | Medium | Hybrid approach, backward compat |

### 6.3 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Key talent departure | Low | High | Knowledge transfer, documentation |
| Supply chain issues (HSM) | Low | Medium | Multi-vendor strategy |
| Quantum timeline acceleration | Low | Critical | Aggressive roadmap execution |

---

## Part 7: Success Metrics

### 7.1 Technical KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| TLS handshake latency | <50ms (99th %ile) | Continuous benchmark |
| V2X verification rate | 1,000+ msg/sec | Field testing |
| Healthcare device battery | <5% crypto overhead | Lab testing |
| Aviation bandwidth efficiency | 80%+ standard PQC | Protocol analysis |
| Safety-critical WCET variance | <10% jitter | Timing analysis |

### 7.2 Business KPIs

| Metric | Target | Timeline |
|--------|--------|----------|
| Domain modules released | 4 | Q4 2026 |
| Enterprise customers | 50+ | Q2 2027 |
| Compliance certifications | 8+ | Q4 2026 |
| Patent applications | 5+ | Q4 2026 |
| Revenue from PQC | $5M+ ARR | Q4 2027 |

### 7.3 Adoption KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| PQC key exchange adoption | 90%+ connections | Telemetry |
| Hybrid certificate deployment | 50%+ customers | Customer survey |
| Domain module usage | 20%+ by sector | Product analytics |

---

## Part 8: Immediate Action Items

### Week 1-2

- [ ] Finalize algorithm selection (ML-KEM-768, Falcon priority)
- [ ] Scope Phase 1 Rust implementation
- [ ] Identify domain specialist partners
- [ ] Initiate FIPS 140-3 pre-assessment

### Week 3-4

- [ ] Begin ML-KEM-768 integration
- [ ] Draft hybrid certificate format based on IETF drafts
- [ ] Establish healthcare partnership discussions
- [ ] Set up automotive V2X test environment

### Month 2

- [ ] Complete ML-KEM-768 implementation
- [ ] Begin Falcon integration
- [ ] Draft domain module architectures
- [ ] Initiate certification readiness assessment

### Month 3

- [ ] Release Phase 1 internal beta
- [ ] Begin healthcare module development
- [ ] Begin automotive module development
- [ ] Complete TLS 1.3 PQC integration

---

## Appendix A: Algorithm Quick Reference

| Algorithm | Type | Public Key | Signature/Ciphertext | Security Level |
|-----------|------|-----------|---------------------|----------------|
| ML-KEM-512 | KEM | 800 B | 768 B | 1 (128-bit) |
| ML-KEM-768 | KEM | 1,184 B | 1,088 B | 3 (192-bit) |
| ML-KEM-1024 | KEM | 1,568 B | 1,568 B | 5 (256-bit) |
| ML-DSA-44 | Sig | 1,312 B | 2,420 B | 2 (128-bit) |
| ML-DSA-65 | Sig | 1,952 B | 3,293 B | 3 (192-bit) |
| ML-DSA-87 | Sig | 2,592 B | 4,595 B | 5 (256-bit) |
| Falcon-512 | Sig | 897 B | 666 B | 1 (128-bit) |
| Falcon-1024 | Sig | 1,793 B | 1,280 B | 5 (256-bit) |
| SLH-DSA-128f | Sig | 32 B | 17,088 B | 1 (128-bit) |
| SLH-DSA-256f | Sig | 64 B | 49,856 B | 5 (256-bit) |

---

## Appendix B: Competitive Positioning

### Market Position

QBITEL differentiates through:

1. **Domain-Specific Optimization**: No competitor offers healthcare/automotive/aviation modules
2. **Agentic AI Integration**: Autonomous security decisions with PQC
3. **Hybrid Maturity**: Production-proven classical + PQC approach
4. **Air-Gapped Capability**: Critical for government/defense
5. **End-to-End Platform**: Discovery → Protection → Compliance

### Competitor Analysis

| Competitor | Strengths | Gaps vs. QBITEL |
|------------|-----------|-----------------|
| PQShield | First FIPS 140-3 cert | No domain optimization |
| ISARA | Strong PKI tools | No AI integration |
| Crypto4A | Crypto-agile HSM | No cloud-native |
| SandboxAQ | Enterprise orchestration | No constrained device support |

---

## Appendix C: Reference Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QBITEL PQC Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Domain Modules Layer                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │Healthcare│  │Automotive│ │Aviation │  │Industrial/OT   │  │   │
│  │  │ Module   │  │ V2X      │ │ Module  │  │ Module         │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Advanced Primitives Layer                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │ PQ-ZKP  │  │ PQ-Creds│  │ PQ-VRF │  │ PQ-FPE          │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Core PQC Engine (Rust)                     │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │ ML-KEM  │  │ ML-DSA  │  │ Falcon │  │ SLH-DSA/LMS    │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │Key Mgmt │  │HSM PKCS#11│ │Hybrid  │  │ Batch Verify   │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Integration Layer                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │   │
│  │  │TLS 1.3  │  │Istio/   │  │ Cloud  │  │ Protocol        │  │   │
│  │  │ PQC     │  │ Envoy   │  │ HSMs   │  │ Discovery       │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0*
*Last Updated: December 2025*
*Next Review: January 2026*
