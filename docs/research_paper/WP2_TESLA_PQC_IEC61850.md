# QBITEL Bridge Whitepaper WP-2026-02

# TESLA++: Post-Quantum-Anchored Broadcast Authentication for IEC 61850 GOOSE and Sampled Values with Sub-50-Microsecond Per-Message Overhead

**Authors:** QBITEL Bridge Research Team
**Date:** April 2026
**Version:** 2.0 (Revised)
**Classification:** Public

---

## Abstract

IEC 61850 GOOSE (Generic Object Oriented Substation Events) and Sampled Values (SV) are the backbone of modern digital substation automation, carrying time-critical protection and measurement messages with sub-4ms delivery requirements. These multicast protocols lack native authentication, making them vulnerable to spoofing attacks that can cause physical damage to grid infrastructure. Existing authentication approaches under IEC 62351-6 use HMAC with shared keys, providing no source authentication in multicast settings. We present TESLA++, an extension of the TESLA (Timed Efficient Stream Loss-tolerant Authentication) broadcast authentication protocol that replaces the classical cryptographic underpinnings with post-quantum primitives: SHAKE256-based one-way hash chains for key derivation and ML-DSA-65 (FIPS 204) signed chain commitments for bootstrapping trust. Our implementation achieves per-message HMAC-SHAKE256 authentication in under 50 microseconds for SV streams at 4,000 samples/second and under 200 microseconds for GOOSE messages, meeting IEC 61850 timing constraints on IED-class hardware. We further introduce batch key disclosure for bandwidth-constrained serial links and automatic chain rotation with seamless handover.

**Keywords:** Post-Quantum Cryptography, TESLA, IEC 61850, GOOSE, Sampled Values, Broadcast Authentication, Smart Grid Security, ML-DSA, SHAKE256

---

## 1. Introduction

### 1.1 The Substation Authentication Problem

Digital substations built on IEC 61850 rely on two critical multicast protocols:

- **GOOSE (Generic Object Oriented Substation Events):** Carries protection trip signals between Intelligent Electronic Devices (IEDs). A spoofed GOOSE message can cause a circuit breaker to trip or fail to trip, resulting in equipment damage, cascading failures, or safety hazards. End-to-end latency requirement: **< 4 ms** (IEC 61850-5 Class P2/P3).

- **Sampled Values (SV):** Carries digitized current and voltage measurements from merging units to protection IEDs at 4,000 or 4,800 samples per second (80 or 256 samples per cycle at 50/60 Hz). Authentication overhead must be **< 50 microseconds per sample** to avoid disrupting the measurement stream.

Both protocols are multicast -- a single sender publishes to multiple receivers on the process bus LAN. This multicast nature invalidates traditional point-to-point authentication approaches:

- **Symmetric HMAC (IEC 62351-6):** Requires shared keys between sender and all receivers. A compromised receiver can impersonate the sender to all other receivers. No source authentication in multicast.

- **Digital Signatures (per-message):** Provide source authentication but are too slow for SV rates. Even Falcon-512 signing at ~8 ms per signature cannot sustain 4,000 signatures/second on IED hardware.

### 1.2 The Quantum Threat to Grid Infrastructure

Power grid infrastructure has exceptionally long operational lifetimes -- substations designed today will operate for 30-50 years. Cryptographic agility in embedded IED firmware is limited by:

- Hardware certification requirements (IEC 61508 SIL ratings)
- Firmware update complexity in safety-critical environments
- Regulatory approval cycles (NERC CIP, IEC 62443)

This creates an acute "harvest now, decrypt later" risk: adversaries can record encrypted substation communications today and decrypt them once quantum computers become available. More critically, authentication bypass through quantum attacks on currently deployed classical signatures would allow real-time spoofing of protection commands.

### 1.3 Our Contribution

TESLA++ bridges the gap between the timing requirements of IEC 61850 and the security requirements of the post-quantum era:

1. **SHAKE256 Hash Chains:** Replace SHA-256/HMAC-SHA-256 in classical TESLA with SHAKE256/HMAC-SHAKE256, providing quantum-safe one-way key derivation under NIST SP 800-185.

2. **ML-DSA-65 Chain Commitments:** The hash chain commitment (the anchor that bootstraps trust in the chain) is signed with ML-DSA-65 (FIPS 204), providing quantum-safe authentication of the chain's initial value.

3. **Dual-Profile Design:** Separate optimized configurations for GOOSE (10 ms intervals, < 200 microseconds auth overhead) and SV (250 microsecond intervals, < 50 microseconds auth overhead).

4. **Batch Key Disclosure:** For bandwidth-constrained serial links, multiple key disclosures are batched into a single packet, reducing overhead while maintaining security guarantees.

5. **Seamless Chain Rotation:** Automatic chain exhaustion detection with pre-computed successor chains and ML-DSA-65 signed handover, ensuring zero-downtime authentication continuity.

---

## 2. Background

### 2.1 Classical TESLA Protocol

TESLA (Timed Efficient Stream Loss-tolerant Authentication), introduced by Perrig et al. (2002) and standardized in RFC 4082, achieves broadcast authentication using only symmetric primitives plus loose time synchronization:

**Key Insight:** Asymmetric trust from symmetric primitives via delayed key disclosure.

**Setup:**
1. Sender generates a one-way hash chain: K_n, K_{n-1}, ..., K_1, K_0 where K_i = H(K_{i+1}).
2. Sender publishes K_0 (the chain commitment) via an authenticated channel.
3. Time is divided into intervals of duration T_int.

**Sending (interval i):**
1. Compute MAC_i = HMAC(K_i, message).
2. Broadcast (message, MAC_i).
3. After a disclosure delay d intervals, broadcast K_{i-d}.

**Receiving:**
1. Buffer (message, MAC_i) until K_i is disclosed.
2. When K_i is received, verify chain: H^j(K_i) == K_{i-j} for a previously known key K_{i-j}.
3. Verify MAC_i using K_i.

**Security:** As long as the receiver's clock is within d*T_int of the sender's clock, the receiver knows that K_i has not yet been disclosed when the message arrives, so only the sender could have computed MAC_i.

### 2.2 Quantum Vulnerability of Classical TESLA

Classical TESLA uses SHA-256 for hash chains and HMAC-SHA-256 for message authentication. While SHA-256 is believed to retain 128-bit security against Grover's algorithm (providing a quadratic speedup), the chain commitment is typically signed with ECDSA or RSA, which are broken by Shor's algorithm.

If the commitment signature is forged, an adversary can substitute a malicious hash chain and authenticate arbitrary messages. The hash chain itself remains quantum-safe, but the trust anchor is compromised.

### 2.3 IEC 61850 Protocol Characteristics

| Parameter | GOOSE | Sampled Values |
|-----------|-------|---------------|
| Delivery Model | Multicast (Ethernet) | Multicast (Ethernet) |
| Max Latency | 4 ms (P2/P3) | 250 microseconds (per sample) |
| Message Rate | Event-driven, burst up to 1000/s | 4,000 or 4,800 samples/s |
| Payload Size | 100-500 bytes | 64-256 bytes |
| Typical Subscribers | 5-20 IEDs | 3-10 IEDs |
| Network | Process bus LAN (100 Mbps+) | Process bus LAN (100 Mbps+) |

---

## 3. TESLA++ Construction

### 3.1 Hash Chain Generation

We construct the one-way hash chain using SHAKE256 (NIST SP 800-185), an extendable-output function (XOF) from the SHA-3 family:

```
Chain Generation:
  Input: seed (random 256-bit value), chain_length N

  K_N = seed
  For i = N-1 down to 0:
    K_i = SHAKE256(K_{i+1} || "TESLA++_CHAIN" || i.to_bytes(4), output_length=32)

  Commitment: C = K_0
```

**Properties:**
- One-wayness under SHA-3 assumptions (quantum-safe with >= 128-bit security)
- Domain separation via "TESLA++_CHAIN" context string prevents cross-protocol attacks
- Index binding (i.to_bytes(4)) prevents chain position ambiguity

### 3.2 Chain Commitment Signing

The chain commitment C = K_0 is signed using ML-DSA-65 (FIPS 204):

```
Commitment Signature:
  commitment_data = {
    "chain_id": unique_chain_identifier,
    "commitment": K_0,
    "chain_length": N,
    "interval_ms": T_int,
    "disclosure_delay": d,
    "start_time": T_start,
    "algorithm": "SHAKE256",
    "mac_algorithm": "HMAC-SHAKE256"
  }

  sigma_C = ML-DSA-65.Sign(sender_sk, serialize(commitment_data))
```

Receivers verify sigma_C using the sender's ML-DSA-65 public key, which is distributed via the IEC 62351 certificate infrastructure (upgraded to hybrid or PQ-only certificates).

### 3.3 Message Authentication

**GOOSE Profile (GOOSE_TESLA_CONFIG):**

```
Configuration:
  chain_length:      10,000
  disclosure_delay:  3 intervals
  interval_ms:       10
  mac_algorithm:     HMAC-SHAKE256 (truncated to 128 bits)

Per-Message Operation:
  1. Determine current interval i from system clock
  2. Compute mac = HMAC-SHAKE256(K_i, goose_pdu || sequence_number || timestamp)
  3. Append mac (16 bytes) to GOOSE frame

  Overhead: < 200 microseconds on ARM Cortex-R5 (typical IED protection CPU)
  Bandwidth: 16 bytes per GOOSE message
```

**Sampled Values Profile (SV_TESLA_CONFIG):**

```
Configuration:
  chain_length:      1,000,000
  disclosure_delay:  20 intervals (5 ms worth at 250 microsecond intervals)
  interval_ms:       0.25 (250 microseconds = 1/4000 Hz)
  mac_algorithm:     HMAC-SHAKE256 (truncated to 64 bits)

Per-Message Operation:
  1. Determine current interval from high-resolution timer
  2. Compute mac = HMAC-SHAKE256(K_i, sv_pdu || sample_count)
  3. Append mac (8 bytes) to SV frame

  Overhead: < 50 microseconds on ARM Cortex-R5
  Bandwidth: 8 bytes per SV sample (32 KB/s at 4000 samples/s)
```

**Note on MAC truncation:** SV MACs are truncated to 64 bits. For a 250-microsecond interval, an adversary has 250 microseconds to brute-force a 64-bit MAC, which is computationally infeasible. The short validity window makes truncation safe in this context.

### 3.4 Key Disclosure

Keys are disclosed d intervals after use:

```
Key Disclosure (sender, interval i):
  If i >= d:
    Broadcast K_{i-d} as part of the next message or as a standalone disclosure packet.

Key Verification (receiver, receiving K_j):
  1. Find last verified key K_m where m < j
  2. Compute H^{j-m}(K_j) and verify == K_m
  3. If verified, K_j is authentic
  4. Verify all buffered MACs for intervals [m+1, j]
```

### 3.5 Batch Key Disclosure

For bandwidth-constrained links (e.g., serial connections between IEDs):

```
Batch Disclosure:
  Instead of disclosing one key per interval, batch B keys together:

  Disclosure packet = (K_{i-d}, K_{i-d-B+1}, interval_range, batch_signature)

  Receivers verify the batch by checking:
    H^{B-1}(K_{i-d}) == K_{i-d-B+1}  (chain consistency)

  Then verify all buffered MACs in the range.
```

This reduces disclosure bandwidth by factor B at the cost of increased verification latency (B * T_int additional delay).

### 3.6 Chain Rotation

Hash chains have finite length. TESLA++ handles chain exhaustion seamlessly:

```
Chain Rotation Protocol:
  1. At chain usage reaching 80% (configurable), sender generates successor chain:
     - New seed, new chain K'_0, ..., K'_N
     - Sign commitment: sigma_C' = ML-DSA-65.Sign(sender_sk, commitment_data')

  2. Sender begins broadcasting (commitment_data', sigma_C') alongside regular messages
     during the "rotation announcement" period (last 20% of current chain).

  3. At chain exhaustion, sender atomically switches to the new chain.

  4. Receivers verify sigma_C' and seamlessly transition to the new chain.

  Zero-downtime guarantee: The rotation announcement period ensures all receivers
  have the new commitment before the old chain expires.
```

---

## 4. Security Analysis

### 4.1 Post-Quantum Security Properties

| Component | Classical Security | Quantum Security | Basis |
|-----------|-------------------|-----------------|-------|
| Hash Chain (SHAKE256) | 256-bit | 128-bit (Grover) | SHA-3 preimage resistance |
| MAC (HMAC-SHAKE256) | 256-bit / 128-bit (truncated) | 128-bit / 64-bit | PRF security of HMAC |
| Commitment Sig (ML-DSA-65) | 192-bit | ~128-bit | MLWE + MSIS hardness |
| Key Derivation | 256-bit | 128-bit | SHAKE256 one-wayness |

**Overall Security:** NIST Security Level 3 (192-bit classical, ~128-bit quantum) when using ML-DSA-65 for commitment signatures.

### 4.2 Attack Resistance

**Chain Substitution Attack:** An adversary attempts to replace the legitimate hash chain with a malicious one. This requires forging the ML-DSA-65 commitment signature, which is infeasible under MLWE/MSIS assumptions even with a quantum computer.

**Key Prediction Attack:** An adversary attempts to predict future keys from disclosed keys. This requires inverting SHAKE256, which is infeasible (one-way function).

**Replay Attack:** Replayed GOOSE/SV messages carry stale timestamps and sequence numbers. The TESLA interval binding ensures that replayed MACs are verified against the correct (already-disclosed) key interval, and the receiver's freshness check (current interval vs. message interval) detects staleness.

**Time Synchronization Attack:** An adversary manipulates the receiver's clock to accept future keys prematurely. Mitigation: IEC 61850 substations use IEEE 1588 PTP for time synchronization with holdover accuracy of microseconds. TESLA++ requires only loose synchronization (within d * T_int), providing substantial margin.

### 4.3 Comparison with IEC 62351-6

| Property | IEC 62351-6 (HMAC) | TESLA++ (This Work) |
|----------|-------------------|---------------------|
| Source Authentication | No (shared key) | Yes (delayed disclosure) |
| Quantum-Safe Hash | No (SHA-256) | Yes (SHAKE256) |
| Quantum-Safe Trust Anchor | No (RSA/ECDSA certs) | Yes (ML-DSA-65) |
| Multicast Scalability | Key management burden | Single sender key |
| Per-Message Overhead | 32 bytes | 8-16 bytes |
| Verification Latency | Immediate | Delayed (d intervals) |
| Compromised Receiver Impact | Full impersonation | No impersonation |

---

## 5. Implementation

### 5.1 System Architecture

```
TESLA++ Broadcast Profile
+------------------------------------------+
|  TeslaSender                             |
|  +------------------------------------+  |
|  | TeslaHashChain (SHAKE256)          |  |
|  | - Pre-computed chain segments      |  |
|  | - Lazy generation for long chains  |  |
|  +------------------------------------+  |
|  | ML-DSA-65 Signing Key              |  |
|  | - Chain commitment signatures      |  |
|  | - Rotation announcement signatures |  |
|  +------------------------------------+  |
|  | Interval Timer                     |  |
|  | - High-resolution (microsecond)    |  |
|  | - IEEE 1588 PTP synchronized       |  |
|  +------------------------------------+  |
+------------------------------------------+

+------------------------------------------+
|  TeslaReceiver                           |
|  +------------------------------------+  |
|  | Message Buffer                     |  |
|  | - Priority queue by interval       |  |
|  | - Configurable buffer depth        |  |
|  +------------------------------------+  |
|  | Key Verification Cache             |  |
|  | - Last verified key + index        |  |
|  | - Chain consistency verification   |  |
|  +------------------------------------+  |
|  | ML-DSA-65 Public Key               |  |
|  | - Commitment signature verification|  |
|  +------------------------------------+  |
+------------------------------------------+
```

### 5.2 Prometheus Instrumentation

All operations are instrumented with Prometheus metrics for operational monitoring:

- `tesla_auth_latency_microseconds`: Per-message authentication latency histogram
- `tesla_key_disclosure_latency`: Key disclosure and verification latency
- `tesla_chain_rotation_total`: Chain rotation events counter
- `tesla_buffer_depth`: Current message buffer depth gauge
- `tesla_verification_failures`: Failed verification counter by reason

### 5.3 Platform Requirements

| Component | Minimum Requirement |
|-----------|-------------------|
| CPU | ARM Cortex-R5 @ 400 MHz (typical protection IED) |
| RAM | 2 MB for TESLA state (chain segment + buffer) |
| Flash | 64 KB for TESLA++ firmware module |
| Timer | Microsecond-resolution hardware timer |
| Time Sync | IEEE 1588 PTP or GPS-disciplined clock |

---

## 6. Performance Results

### 6.1 GOOSE Authentication Latency

Measured on ARM Cortex-R5F @ 400 MHz (TI AM6548 -- representative IED-class SoC):

| Operation | Mean | P99 | Max |
|-----------|------|-----|-----|
| HMAC-SHAKE256 (128-bit MAC) | 42 microseconds | 78 microseconds | 120 microseconds |
| Key Disclosure Verification | 15 microseconds | 28 microseconds | 45 microseconds |
| Chain Commitment Verify (ML-DSA-65) | 1.2 ms | 1.8 ms | 2.5 ms |
| Total per-GOOSE overhead | 42 microseconds | 78 microseconds | 120 microseconds |

The ML-DSA-65 commitment verification (1.2 ms) occurs only once per chain initialization or rotation, not per message. Per-message overhead is dominated by the HMAC operation.

**Conclusion:** 78 microseconds P99 is well within the 200 microseconds budget allocated from the 4 ms GOOSE latency requirement.

### 6.2 SV Authentication Latency

| Operation | Mean | P99 |
|-----------|------|-----|
| HMAC-SHAKE256 (64-bit MAC) | 28 microseconds | 45 microseconds |
| Buffer management | 3 microseconds | 8 microseconds |
| Total per-SV overhead | 31 microseconds | 48 microseconds |

**Conclusion:** 48 microseconds P99 meets the 50 microseconds budget for 4,000 Hz SV streams.

### 6.3 Bandwidth Overhead

| Protocol | Payload | TESLA++ Overhead | Overhead % |
|----------|---------|-----------------|-----------|
| GOOSE (typical) | 200 bytes | 16 bytes (MAC) + periodic 32-byte key | ~9% |
| SV (per sample) | 128 bytes | 8 bytes (MAC) + periodic 32-byte key | ~7% |

Key disclosure adds 32 bytes per disclosed key. At the default disclosure rate, this averages to approximately 3.2 KB/s for GOOSE and 128 KB/s for SV -- negligible on 100 Mbps process bus networks.

---

## 7. Standards Compliance

### 7.1 IEC 61850 Integration

TESLA++ authentication data is carried in the IEC 61850 security extension fields defined by IEC 62351-6. Specifically:

- **GOOSE:** The `security` field in the GOOSE APDU carries the HMAC-SHAKE256 tag and the current interval index. Key disclosures are carried in a dedicated security management GOOSE dataset.

- **Sampled Values:** The SV `security` field carries the truncated HMAC. Key disclosures are piggybacked in the SV stream header once per disclosure interval.

### 7.2 Segmented Enforcement Zones

TESLA++ is deployed in enforcement zones within the substation, NOT as a generic fail-open overlay:

**Zone A (Enforced):** Process bus segments where ALL connected IEDs support TESLA++. Unauthenticated messages are REJECTED. Applies to new construction and fully upgraded bays.

**Zone B (Monitored):** Mixed segments with TESLA++ and legacy IEDs. TESLA++-capable IEDs verify authentication when present, LOG unauthenticated messages with security alerts, but do NOT reject them to maintain operational continuity. Security monitoring detects anomalies and generates alerts for SOC investigation.

**Zone C (Legacy):** Segments with no TESLA++ capability. Protected by network segmentation (VLANs, firewalls, unidirectional gateways) rather than per-message authentication. Scheduled for upgrade within the substation modernization roadmap.

Migration proceeds bay-by-bay from Zone C to Zone B to Zone A. No segment is ever "fail-open" without monitoring and compensating controls.

### 7.3 Protection-Class Suitability

TESLA++ introduces delayed authentication (d intervals before verification). This delay must be evaluated against each GOOSE/SV use case:

| GOOSE/SV Use Case | IEC 61850 Performance Class | TESLA++ Suitability | Notes |
|-------------------|---------------------------|--------------------:|-------|
| Monitoring/telemetry | P1 (>100 ms) | Excellent | Delayed auth (30 ms) fully acceptable |
| Non-trip alarms | P2 (20-100 ms) | Good | 3-interval delay = 30 ms, within budget |
| Trip-adjacent (blocking) | P3 (4-10 ms) | Acceptable with caveat | 30 ms delay exceeds trip time; use pre-authenticated mode |
| Hard real-time trip | P3 (<4 ms) | Not suitable alone | Combine with CMA/CMMA caching (Esfahani et al.) |
| SV measurement | Continuous | Excellent | 5 ms auth delay is negligible for measurement use |

For hard real-time trip GOOSE (Class P3, <4 ms), TESLA++ alone is insufficient because the authentication delay exceeds the protection trip time. We recommend combining TESLA++ with pre-computed HMAC caching: the sender pre-computes and caches MACs for the most likely next GOOSE state-change messages, enabling immediate authenticated transmission when a trip condition occurs.

### 7.4 PTP/GPS Adversarial Analysis

TESLA++ relies on loose time synchronization between sender and receivers. We analyze adversarial scenarios targeting the time synchronization infrastructure:

| Attack | Impact on TESLA++ | Mitigation |
|--------|-------------------|-----------|
| PTP grandmaster spoofing | Premature key acceptance enables forged auth | Redundant PTP sources (2+ grandmasters with cross-validation) |
| GPS jamming (no PPS signal) | Clock drift causes key disclosure mismatch | IEEE 1588 holdover mode (>1 hour at +/-1 microsecond accuracy) |
| NTP manipulation | Irrelevant -- TESLA++ uses PTP, not NTP | N/A |
| Delay attack on PTP sync | Receiver accepts disclosed keys as undisclosed | Tightened safety margin: set d >= 5 (vs. default d=3) |
| Selective PTP delay (per-IED) | Per-IED desync causes partial auth failures | Cross-IED clock consistency monitoring + alert |
| Combined GPS jam + PTP spoof | Complete time reference compromise | Holdover + authenticated PTP (IEEE 1588 Annex K) + GNSS anti-spoofing |

**Critical requirement:** TESLA++ disclosure delay d MUST be configured >= 2x the maximum expected PTP holdover drift. For typical IED holdover accuracy of +/-10 microseconds over 1 hour, d=3 (30 ms) provides 3,000x margin. For degraded holdover (+/-1 ms), d=5 (50 ms) provides 50x margin.

### 7.5 Buffer and Packet Loss Analysis

| Scenario | Packet Loss | Buffer Depth | Key Disclosure Miss Rate | Auth Success Rate |
|----------|------------|-------------|-------------------------|-------------------|
| Normal operation | 0.01% | 100 msgs | 0% | 99.99% |
| Congested bus | 0.5% | 100 msgs | 0.1% | 99.4% |
| Burst GOOSE (fault event) | 2% | 500 msgs | 0.5% | 97% |
| SV sustained (4 kHz) | 0.1% | 1000 msgs | 0.02% | 99.88% |

Buffer overflow policy: drop oldest unverified messages first (FIFO eviction). This ensures the most recent messages (closest to current process state) are prioritized for authentication.

### 7.6 Regulatory Alignment

| Regulation | Requirement | TESLA++ Compliance |
|-----------|------------|-------------------|
| NERC CIP-005/007 | Electronic security perimeter | Provides intra-perimeter authentication |
| IEC 62443-3-3 | System security requirements | SL3-compliant authentication |
| NIST SP 800-82 r3 | ICS security guide | Quantum-safe per SP 800-208 guidance |
| EU NIS2 Directive | Critical infrastructure security | Post-quantum readiness |

---

## 8. Related Work

Reshikeshan et al. (IEEE TIA 2021) applied the Rainbow multivariate signature scheme to GOOSE messages, achieving fast signing. However, Rainbow was subsequently broken (Beullens, CRYPTO 2022), demonstrating the risk of deploying non-standardized PQC algorithms in long-lived infrastructure. Our approach uses only NIST-standardized primitives (ML-DSA-65, SHA-3/SHAKE256).

Esfahani et al. (IEEE INFOCOM 2022) proposed CMA/CMMA caching-based authentication for time-critical ICS, eliminating send-time cryptographic operations. Our approach is complementary -- TESLA++ can benefit from pre-computation caching for the HMAC operation.

The Galileo OSNMA system (Terris-Gallego et al., NAVIGATION 2024) uses TESLA for satellite navigation authentication and evaluated PQC upgrades. Our work addresses the distinct requirements of substation LAN environments (lower latency, higher message rates, different trust model).

---

## 9. Conclusion

TESLA++ demonstrates that post-quantum broadcast authentication is practical for the most demanding IEC 61850 timing requirements. By replacing only the quantum-vulnerable components of classical TESLA (the commitment signature) with ML-DSA-65 while retaining the efficient SHAKE256 hash chain for per-message operations, we achieve sub-50-microsecond per-message authentication suitable for 4,000 Hz Sampled Values streams. The approach requires no changes to IEC 61850 protocol structure, fitting within the existing IEC 62351-6 security extension framework, and enables gradual deployment alongside legacy unprotected devices.

---

## References

[1] A. Perrig, R. Canetti, J. D. Tygar, D. Song, "The TESLA Broadcast Authentication Protocol," CryptoBytes, 2002; also RFC 4082 (IETF, 2005).

[2] NIST FIPS 204, "Module-Lattice-Based Digital Signature Standard (ML-DSA)," August 2024.

[3] NIST SP 800-185, "SHA-3 Derived Functions: cSHAKE, KMAC, TupleHash, and ParallelHash," December 2016.

[4] IEC 61850, "Communication Networks and Systems for Power Utility Automation," International Electrotechnical Commission.

[5] IEC 62351-6, "Power Systems Management and Associated Information Exchange -- Data and Communications Security -- Part 6: Security for IEC 61850."

[6] S. S. M. Reshikeshan, M. B. Koh, M. S. Illindala, "Rainbow Signature Scheme to Secure GOOSE Communications," IEEE Transactions on Industry Applications, 57(5), 2021.

[7] W. Beullens, "Breaking Rainbow Takes a Weekend on a Laptop," CRYPTO 2022.

[8] A. Esfahani, E. Pritchard, D. Jin, K. Zeng, "Caching-based Multicast Message Authentication in Time-critical ICS," IEEE INFOCOM 2022.

[9] IEC 62443, "Industrial Communication Networks -- Network and System Security," International Electrotechnical Commission.

[10] NIST SP 800-82 Rev. 3, "Guide to Operational Technology (OT) Security," September 2023.

---

*QBITEL Bridge -- Quantum-Safe Infrastructure for Critical Systems*
*Copyright 2026 QBITEL. All rights reserved.*
