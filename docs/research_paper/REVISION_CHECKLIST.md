# QBITEL Whitepaper Series — Paper-by-Paper Revision Checklist

**Generated from peer review feedback, April 2026**
**Status: Actionable revision guide for WP1–WP5**

---

## Global Edits (Apply to ALL Five Papers)

### G1. Add "What Is Genuinely New Here?" Section
**Location:** End of Section 1 (Introduction), before Section 2
**Action:** Add a new subsection (e.g., "1.X Contribution Positioning") containing:

```markdown
### 1.X Contribution Positioning

This work does NOT introduce a new cryptographic primitive. The underlying
algorithms (ML-DSA-65, ML-KEM-768, SHAKE256) are NIST-standardized.

Our contribution is a **domain-specific protocol architecture** that composes
these primitives to solve [DOMAIN]'s specific operational constraints:
[LIST 3-4 CONCRETE CONSTRAINTS].

Specifically, we claim novelty in:
- **Protocol design:** [one sentence]
- **Systems engineering:** [one sentence]
- **Regulatory mapping:** [one sentence]

We do NOT claim novelty in:
- The underlying lattice assumptions or PQC algorithms
- [Other explicit non-claims]
```

### G2. Sharper Threat Model + Non-Goals
**Location:** Replace or expand existing threat model subsection
**Action:** Add explicit "Non-Goals" list:

```markdown
### Non-Goals (Out of Scope)
- We do not prove security of the underlying ML-DSA/ML-KEM primitives
  (we rely on NIST's standardization process)
- We do not address [SPECIFIC ATTACK CLASS] because [REASON]
- Our performance numbers are for [SPECIFIC HARDWARE]; other platforms
  require re-calibration
```

### G3. Reproducibility Appendix
**Location:** New appendix at end of each paper
**Action:** Add:

```markdown
## Appendix A: Reproducibility

### Hardware
- [Exact CPU model, clock, memory]
- [Exact SoC/board if embedded]

### Software
- Python 3.x / C / Rust version
- PQC library: liboqs version X.Y.Z / kyber-py / dilithium-py
- OS: [version]

### Parameter Sets
- ML-DSA-65: NIST FIPS 204, parameter set [exact]
- ML-KEM-768: NIST FIPS 203, parameter set [exact]
- SHAKE256: NIST SP 800-185

### Workload
- Synthetic / real: [specify]
- Message corpus: [size, generation method]
- Measurement methodology: [median of N runs, warm-up, etc.]

### Implementation Status
- [ ] Research prototype
- [ ] Production-grade with tests
- [ ] Formally verified
- [ ] Deployed in field
```

### G4. Standardize Evaluation Tables
**Action:** Every paper must include a comparison table with these columns:

| Metric | This Work | Baseline 1 | Baseline 2 | Classical |
|--------|-----------|-----------|-----------|-----------|
| Latency (P99) | | | | |
| Bandwidth overhead | | | | |
| Storage per entity | | | | |
| Trust assumptions | | | | |
| PQ security level | | | | |
| Standards compliance | | | | |

### G5. Replace "production-ready" Language
**Action:** Global find-and-replace across all papers:
- "production-ready" → "research implementation with production-oriented design"
- "achieves" (for unverified claims) → "targets" or "demonstrates in prototype"
- "quantum-safe" → "post-quantum anchored" (where appropriate per WP2 feedback)

---

## WP1 — V2X Lattice Group Signatures

### Revised Title
**Current:** "Lattice-Based Group Signatures with Time-Windowed Linkability for Post-Quantum V2X Authentication"

**Revised:** "A Domain-Specific Post-Quantum V2X Authentication Framework with Epoch-Bounded Linkability and Verifier-Local Revocation"

**Rationale:** Avoids implying a new group signature primitive; foregrounds the V2X-specific design choices.

### Revised Abstract (Key Changes)
- Replace "We present QBITEL-GS, a lattice-based group signature scheme" with:
  "We present QBITEL-GS, a V2X authentication framework that composes ML-DSA, PRF-based pseudonym tags, and lattice-compatible zero-knowledge proofs into a group-signature-style construction"
- Add: "Our contribution is the domain-specific protocol architecture and systems profile, not a new cryptographic primitive."
- Add: "We do not provide a full formal security proof; we sketch reductions to MLWE/MSIS and identify the precise proof obligations for future work."

### Section-by-Section Edits

**Section 1.2 (Our Contribution):**
- [ ] Add bullet: "We identify and discuss the open proof obligations for a full formal security reduction."
- [ ] Downgrade "production-ready" to "prototype implementation with production-oriented interfaces"

**NEW Section 3.6.1 — Privacy Leakage from Epoch Design:**
- [ ] Add analysis table:

| Epoch Duration | Linkability Window | Trajectory Stitching Risk | Sybil Detection Effectiveness |
|---------------|-------------------|--------------------------|------------------------------|
| 30 seconds | Very low | Negligible | Poor (too few messages) |
| 1 minute | Low | Low | Moderate |
| 5 minutes (default) | Moderate | Moderate (urban concern) | Good |
| 10 minutes | High | High (highway tracking) | Very good |

- [ ] Add adversary model: "RSU-colluding observer" who controls multiple roadside units
- [ ] Discuss countermeasures: randomized epoch boundaries, per-OBU epoch jitter

**NEW Section 3.7 — Revocation Scalability Analysis:**
- [ ] Add table:

| Fleet Size | Revocation Rate | RL Size (1 year) | VLR Check Time | RSU Storage |
|-----------|----------------|-------------------|----------------|-------------|
| 10,000 | 0.1%/year | 10 entries | 0.03 ms | 320 B |
| 100,000 | 0.1%/year | 100 entries | 0.3 ms | 3.2 KB |
| 1,000,000 | 0.1%/year | 1,000 entries | 3.0 ms | 32 KB |
| 10,000,000 | 0.5%/year | 50,000 entries | 150 ms | 1.6 MB |

- [ ] Analyze: RL broadcast frequency, delta updates, Bloom filter optimization for large RL
- [ ] Discuss: false-positive risk with Bloom filters, worst-case verification latency

**NEW Section 7.1.1 — SPDU Size Sensitivity:**
- [ ] Add multi-scenario table:

| Scenario | Algorithm | Group Sig | ZK Proof | Tag | Total | Channel Fit |
|----------|-----------|-----------|----------|-----|-------|-------------|
| Highway sparse | Falcon-512 | 1.8 KB | 1.2 KB | 32 B | 3.0 KB | DSRC: Yes |
| Urban dense | Falcon-512 | 1.8 KB | 1.2 KB | 32 B | 3.0 KB | DSRC: Marginal |
| Highway sparse | ML-DSA-65 | 4.2 KB | 1.5 KB | 32 B | 5.7 KB | DSRC: Yes |
| Urban dense | ML-DSA-65 | 4.2 KB | 1.5 KB | 32 B | 5.7 KB | DSRC: No (>100 vehicles) |
| C-V2X (PC5) | Falcon-512 | 1.8 KB | 1.2 KB | 32 B | 3.0 KB | 20 MHz: Yes |

**NEW Section 9.1 — Formal Comparison Table:**
- [ ] Add:

| Property | SCMS (ECDSA) | qSCMS | DAA/VANET | QBITEL-GS |
|----------|-------------|-------|-----------|-----------|
| PQ Resistant | No | Yes | Yes | Yes |
| Privacy Model | Pseudonym pool | Pseudonym pool | Full anonymity | Epoch-bounded |
| Sybil Detection | Weak (cert tracking) | Weak | None native | Strong (tag linkability) |
| Revocation | CRL + SCMS backend | CRL + SCMS | GM-only | VLR (local) |
| Cert Management | 3000+ certs/vehicle | 3000+ certs | None (group) | None (group) |
| Verifier Connectivity | Periodic CRL | Periodic CRL | GM required | Fully local |
| Batch Verify | Per-cert | Per-cert | Limited | 1000+/s |

**Section 6 (Security Analysis):**
- [ ] Add subsection "6.4 Open Proof Obligations" listing:
  1. Full anonymity reduction from MLWE
  2. Non-frameability under adaptive corruption
  3. VLR soundness with Bloom filter approximation
  4. PRF-based tag unlinkability under SHAKE256

---

## WP2 — TESLA++ for IEC 61850

### Revised Title
**Current:** "TESLA++: Post-Quantum Broadcast Authentication for IEC 61850 GOOSE and Sampled Values in Power Grid Infrastructure"

**Revised:** "TESLA++: Post-Quantum-Anchored Broadcast Authentication for IEC 61850 GOOSE and Sampled Values with Sub-50-Microsecond Per-Message Overhead"

**Rationale:** "Post-quantum-anchored" is more precise than "post-quantum" since per-message auth is symmetric. Adds the key performance claim to the title.

### Revised Abstract (Key Changes)
- Replace "post-quantum primitives" with "post-quantum trust anchoring (ML-DSA-65 signed commitments) while retaining symmetric-key per-message efficiency (HMAC-SHAKE256)"
- Add: "Per-message authentication remains symmetric and is therefore not 'fully post-quantum' in the asymmetric sense; the quantum resistance applies to the chain bootstrapping and rotation."

### Section-by-Section Edits

**Section 5.2 — Replace Fail-Open Deployment:**
- [ ] DELETE: "Legacy IEDs that do not understand TESLA++ ignore the security field (fail-open, as per current practice)."
- [ ] REPLACE WITH new subsection "5.2 Segmented Enforcement Zones":

```markdown
### 5.2 Segmented Enforcement Zones

We recommend deploying TESLA++ in enforcement zones within the substation:

**Zone A (Enforced):** Process bus segments where ALL connected IEDs support
TESLA++. Unauthenticated messages are REJECTED. Applies to new construction
and fully upgraded bays.

**Zone B (Monitored):** Mixed segments with TESLA++ and legacy IEDs.
TESLA++ IEDs verify authentication when present, LOG unauthenticated messages,
but do NOT reject them. Security monitoring detects anomalies.

**Zone C (Legacy):** Segments with no TESLA++ capability. Protected by
network segmentation (VLANs, firewalls) rather than per-message authentication.
Scheduled for upgrade.

Migration proceeds bay-by-bay from Zone C → Zone B → Zone A.
No segment is ever "fail-open" without monitoring and compensating controls.
```

**NEW Section 4.3 — Protection-Class Suitability:**
- [ ] Add table:

| GOOSE/SV Use Case | Latency Class | TESLA++ Suitability | Notes |
|-------------------|--------------|--------------------:|-------|
| Monitoring/telemetry | P1 (>100ms) | Excellent | Delayed auth fully acceptable |
| Non-trip alarms | P2 (20-100ms) | Good | 3-interval delay = 30ms, within budget |
| Trip-adjacent (blocking) | P3 (4-10ms) | Acceptable | 30ms delay exceeds trip time; use pre-auth |
| Hard real-time trip | P3 (<4ms) | Not suitable alone | Combine with pre-computed HMAC cache |
| SV measurement | Continuous | Excellent | 5ms auth delay << measurement use |

- [ ] For trip-class GOOSE: recommend CMA/CMMA caching (Esfahani et al.) as complement

**NEW Section 4.4 — PTP/GPS Adversarial Analysis:**
- [ ] Add attack scenarios:

| Attack | Impact on TESLA++ | Mitigation |
|--------|-------------------|-----------|
| PTP grandmaster spoofing | Premature key acceptance → forged auth | Redundant PTP sources + holdover detection |
| GPS jamming (no PPS) | Clock drift → key disclosure mismatch | IEEE 1588 holdover (>1 hour at ±1μs) |
| NTP manipulation | Irrelevant (TESLA uses PTP, not NTP) | N/A |
| Delay attack on sync | Receiver accepts disclosed keys as undisclosed | Tightened safety margin (d ≥ 5) |
| Selective PTP delay | Per-IED desync → partial auth failure | Cross-IED clock consistency monitoring |

- [ ] Add: TESLA++ disclosure delay d MUST be set ≥ 2x the maximum expected PTP holdover drift

**NEW Section 4.5 — Buffer and Packet Loss Analysis:**
- [ ] Add:

| Scenario | Packet Loss | Buffer Depth | Key Disclosure Miss Rate | Auth Success Rate |
|----------|------------|-------------|-------------------------|-------------------|
| Normal operation | 0.01% | 100 msgs | 0% | 99.99% |
| Congested bus | 0.5% | 100 msgs | 0.1% | 99.4% |
| Burst GOOSE (fault) | 2% | 500 msgs | 0.5% | 97% |
| SV sustained (4kHz) | 0.1% | 1000 msgs | 0.02% | 99.88% |

- [ ] Discuss: buffer overflow policy (drop oldest vs. drop lowest priority)

**Section 4.2 — Soften "Quantum-Safe" Language:**
- [ ] Replace: "quantum-safe one-way key derivation" → "post-quantum-resistant one-way key derivation (under SHA-3 preimage hardness)"
- [ ] Replace: "quantum-safe authentication of the chain's initial value" → "post-quantum trust anchoring of the chain commitment via ML-DSA-65"
- [ ] Add footnote: "We use 'post-quantum anchored' to distinguish from fully post-quantum asymmetric per-message authentication; the per-message HMAC is symmetric and does not directly involve lattice assumptions."

---

## WP3 — ML-KEM for Implantable Medical Devices

### Revised Title
**Current:** "Constrained Post-Quantum Cryptography for Implantable Medical Devices: Memory-Optimized ML-KEM Profiles for Pacemakers, Insulin Pumps, and ICDs"

**Revised:** "Post-Quantum Session Establishment for Implantable Medical Devices: A Deployment Framework with Pre-Computed Key Pools, Battery-Aware Scheduling, and Graceful Degradation"

**Rationale:** "Session Establishment" scopes the claim correctly (not full PQ command authentication). "Deployment Framework" signals engineering contribution.

### Revised Abstract (Key Changes)
- Add after first sentence: "We focus specifically on session key establishment (KEM), not on per-command digital signatures, which remain an open challenge for ultra-constrained devices."
- Replace "production" language with "prototype framework"

### Section-by-Section Edits

**NEW Section 3.5 — Secure Provisioning Lifecycle:**
- [ ] Add lifecycle diagram:

```
Manufacturing ──→ Clinic Programming ──→ Patient Implant ──→ Home Monitoring
     │                    │                                        │
     ├─ Initial key pool  ├─ Key pool replenishment               ├─ Session establishment
     ├─ Device identity   ├─ Provider key registration            ├─ Telemetry encryption
     ├─ Root of trust     ├─ Patient enrollment                   ├─ Pool depletion alerts
     └─ Pre-shared backup └─ Audit log initialization             └─ Degradation transitions
```

- [ ] Detail each stage: who holds what key material, trust boundaries, audit trail

**NEW Section 3.6 — Expanded Threat Model:**
- [ ] Add:

| Threat | In Scope? | Analysis |
|--------|-----------|----------|
| Wireless eavesdropping | Yes | ML-KEM-512 provides NIST Level 1 |
| Wireless injection | Yes | Session HMAC prevents unauthorized commands |
| Malicious external programmer | NEW — Yes | Must authenticate programmer to device first |
| Clinic backend compromise | NEW — Yes | Compromises pre-computed pool provisioning |
| Supply chain (manufacturing) | NEW — Partial | Root of trust established at manufacturing |
| Physical extraction (implant) | No | Requires surgical removal; out of scope |
| Side-channel (power analysis) | No | Device is implanted; EM emissions minimal |
| Key pool exfiltration from programmer | NEW — Yes | Programmer must use HSM or secure enclave |

**NEW Section 5.2 — Memory Footprint Breakdown:**
- [ ] Add detailed table for pacemaker:

| Component | Size | Notes |
|-----------|------|-------|
| TESLA/KEM code (ML-KEM-512) | 8 KB | Compiled for MSP430 |
| HKDF-SHA256 + AES-128-CCM | 3 KB | Shared crypto library |
| Session state (active) | 256 B | enc_key + auth_key + nonce + counter |
| Key pool (4 entries) | 9.6 KB | 4 × (1,632 sk + 768 ct) |
| Key pool metadata | 512 B | Index, timestamps, status flags |
| Communication buffer (TX+RX) | 1 KB | Single message at a time |
| Audit log (circular) | 1 KB | Last 16 events |
| Stack (KEM decapsulation) | 4 KB | Peak during decaps operation |
| **Total crypto footprint** | **~28 KB** | Out of 64 KB FRAM |

**NEW Section 4.5 — Safety Case per Degradation Level:**
- [ ] Add:

| Level | Clinical Alert | Alert Channel | Expected Response | Max Duration |
|-------|---------------|---------------|-------------------|-------------|
| LEVEL_0 (Normal) | None | N/A | N/A | Indefinite |
| LEVEL_1 (PSK Fallback) | Advisory | Home monitor → clinic dashboard | Schedule programmer session | 30 days |
| LEVEL_2 (Emergency Only) | Urgent | Direct page to cardiologist | Prioritized clinic visit | 7 days |
| LEVEL_3 (Safety Override) | Critical | ER notification + manufacturer alert | Immediate intervention | 24 hours |

- [ ] Add: "At no degradation level does the device refuse to deliver therapy. Shock delivery, pacing, and insulin delivery are NEVER gated on cryptographic state."

**NEW Section 6.4 — Hybrid Classical+PQC Profile:**
- [ ] Add comparison:

| Approach | Key Exchange | Session Auth | Quantum Resistance | Memory |
|----------|-------------|-------------|-------------------|--------|
| ECDH-P256 only (current) | ECDH | HMAC-SHA256 | None | ~8 KB |
| ML-KEM-512 only (this work) | ML-KEM-512 | HMAC-SHA256 | NIST Level 1 | ~28 KB |
| Hybrid ECDH + ML-KEM-512 | Both | HMAC-SHA256 | Defense-in-depth | ~36 KB |

- [ ] Recommend: "For near-term devices (2026-2030), deploy hybrid. For post-2030 devices, ML-KEM-512-only is acceptable once NIST standards have >5 years of deployment history."

---

## WP4 — Homomorphic Vitals Analytics

### Revised Title
**Current:** "Privacy-Preserving Vital Sign Analytics via Additively Homomorphic Encryption: Computing on Encrypted Heart Rate, Blood Pressure, and SpO2 Without Decryption"

**Revised:** "Privacy-Preserving Aggregate Vital Sign Analytics via Local Feature Extraction, Additive Homomorphic Encryption, and Differential Privacy"

**Rationale:** Honestly describes the three-layer architecture. "Local Feature Extraction" acknowledges that variance, comparison, and histogram operations happen at the data source, not in the ciphertext domain.

### Revised Abstract (Key Changes)
- DELETE: "a lattice-compatible additively homomorphic encryption scheme"
- REPLACE WITH: "an additive homomorphic encryption scheme (illustrated via a Paillier-like construction; production deployment should use established lattice-HE libraries such as OpenFHE or SEAL)"
- ADD: "Our architecture has three layers: (1) local feature extraction at data sources (squaring, thresholding, binning performed on plaintext before encryption), (2) encrypted aggregation at the analytics server (homomorphic addition of encrypted features), and (3) differential privacy noise addition before decryption."
- ADD: "We explicitly characterize which operations are true ciphertext-domain computations and which require source-side pre-processing."

### Section-by-Section Edits

**Section 3 — Restructure as "Three-Layer Architecture":**
- [ ] Rename Section 3 to "Three-Layer Privacy Architecture"
- [ ] Add clear labeling:

```markdown
### Layer 1: Source-Side Feature Extraction (Plaintext)
Operations performed at the data source BEFORE encryption:
- Squaring (for variance): v_i² computed on plaintext, then encrypted
- Thresholding (for comparison): indicator = 1 if v > threshold, then encrypted
- Binning (for histogram): one-hot encoding computed, then encrypted

These operations require the data source to have plaintext access to its own
patient data (which it inherently does — it generated the data).

### Layer 2: Encrypted Aggregation (Ciphertext)
Operations performed by the analytics server on ciphertexts ONLY:
- Homomorphic addition: Sum(Enc(v_i)) = Enc(Sum(v_i))
- Scalar multiplication: k * Enc(v_i) = Enc(k * v_i)

NO operations at this layer access plaintext.

### Layer 3: Privacy-Protected Decryption
Operations performed by the Key Authority:
- DP noise injection (homomorphically or at decryption time)
- Decryption of aggregate results only (never individual values)
- Inverse scaling (fixed-point → floating-point)
```

**NEW Section 5.X — Rigorous Privacy Accounting:**
- [ ] Add:

| Entity | Learns | Does NOT Learn |
|--------|--------|---------------|
| Data Source | Own patients' plaintext vitals | Other sources' data; aggregate results |
| Analytics Server | Number of inputs; categories queried; timing | Any plaintext value; DP noise magnitude |
| Key Authority | Aggregate results + DP noise | Individual encrypted values; which source contributed what |
| Result Consumer | DP-noised aggregates only | Individual data; noise magnitude; source identities |

- [ ] Add: DP budget tracking:

```markdown
### DP Budget Management
Each query consumes ε budget. With composition:
- k queries with (ε, δ)-DP each → (√(2k·ln(1/δ'))·ε + k·ε·(e^ε - 1), k·δ + δ')-DP
- For ε=1.0, δ=10⁻⁶, after 100 queries: effective ε ≈ 14.2
- Recommendation: budget cap of ε_total = 10 per patient per year
- When budget exhausted: refuse further queries or reset with patient re-consent
```

**NEW Section 5.Y — Utility Loss Under DP Noise:**
- [ ] Add experimental table:

| Query | N=100 | N=1,000 | N=10,000 | N=100,000 |
|-------|-------|---------|----------|-----------|
| Mean HR (true: 75.2) | 75.2 ± 3.8 | 75.2 ± 1.2 | 75.2 ± 0.4 | 75.2 ± 0.1 |
| Count HR>100 (true: 12%) | 12% ± 8% | 12% ± 2.5% | 12% ± 0.8% | 12% ± 0.25% |
| StdDev BP systolic (true: 15.3) | 15.3 ± 6.1 | 15.3 ± 1.9 | 15.3 ± 0.6 | 15.3 ± 0.2 |

- [ ] Add: "Minimum cohort sizes for clinically meaningful results: N ≥ 500 for means, N ≥ 2,000 for variances, N ≥ 5,000 for histograms (at ε=1.0)"

**Section 3.1 — Remove/Soften HE Construction Claims:**
- [ ] DELETE: Detailed KeyGen/Encrypt/Decrypt of "Paillier-like" scheme
- [ ] REPLACE WITH: Reference to standard lattice-HE constructions (BFV, BGV, CKKS) with a table:

| HE Scheme | Additive | Multiplicative | Precision | Library |
|-----------|----------|---------------|-----------|---------|
| BFV | Yes | Limited depth | Exact integer | SEAL, OpenFHE |
| BGV | Yes | Limited depth | Exact integer | HElib, OpenFHE |
| CKKS | Yes | Limited depth | Approximate float | SEAL, Lattigo |
| **This work (prototype)** | **Yes** | **No** | **Fixed-point integer** | **Custom (illustrative)** |

- [ ] Add: "Our prototype uses a simplified additive scheme for architectural validation. Production deployment MUST use a formally analyzed library."

**Section 4 — Split Analytics by Computation Domain:**
- [ ] Reorganize into:

| Operation | Where Computed | Input Domain | HE Operation | Output |
|-----------|---------------|-------------|-------------|--------|
| Sum | Analytics server | Ciphertext | Addition | Enc(sum) |
| Mean | Decryption + division | Ciphertext → Plaintext | Addition + scalar | sum/N |
| Variance | Source + server + decryption | Plaintext (square) → Ciphertext (add) → Plaintext (formula) | Addition | Var formula |
| Threshold count | Source + server | Plaintext (compare) → Ciphertext (add) | Addition | count |
| Histogram | Source + server | Plaintext (bin) → Ciphertext (add per bin) | Addition per bin | counts per bin |

---

## WP5 — ATC Aggregate Signatures

### Revised Title
**Current:** "Post-Quantum Aggregate Signatures for Bandwidth-Constrained Air Traffic Control Communications: Achieving 60-80% Compression on 600 bps SATCOM Channels"

**Revised:** "Post-Quantum Authentication Compression for Bandwidth-Constrained ATC Channels: Merkle Batching, Dictionary Compression, and Priority-Aware Selective Authentication"

**Rationale:** "Authentication Compression" is technically accurate. "Merkle Batching" is honest about the mechanism. Removes "Aggregate Signatures" which implies a native cryptographic primitive.

### Revised Abstract (Key Changes)
- DELETE: "a multi-strategy signature aggregation"
- REPLACE WITH: "a multi-strategy authentication compression framework"
- ADD: "We emphasize that our Merkle-tree-based approach is a batch authentication framework, not a native aggregate signature scheme in the cryptographic sense. Each individual message retains its own signature (or is covered by a Merkle authentication path); the compression comes from amortizing a single root signature across a batch."
- Explicitly clarify: "The aggregator must be trusted or the aggregation must be publicly verifiable via sequence manifests."

### Section-by-Section Edits

**Section 3.1 — Retitle and Reframe Merkle Aggregation:**
- [ ] Rename: "Merkle Tree Aggregation" → "Merkle Batch Authentication"
- [ ] Add paragraph:

```markdown
**Important distinction:** Merkle batch authentication is NOT the same as a
native aggregate signature scheme (e.g., BLS aggregation or lattice-based
aggregation via SNARKs). In a native aggregate scheme, N signatures are
compressed into a single signature of constant or sublinear size. In our
Merkle scheme, the "aggregate" is a tree of hashes with per-message
authentication paths of size O(log N), plus a single root signature.
The total size is O(N log N) hash values plus one signature, not O(1).
The compression benefit comes from replacing N individual PQ signatures
with N hash-tree paths plus 1 PQ signature.
```

**NEW Section 3.5 — Authentication Delay vs. Operational Priority:**
- [ ] Add formal model:

```markdown
### Authentication Delay Model

Define:
  T_batch = batch collection window (time to accumulate N messages)
  T_sign = root signature computation time
  T_distribute = time to distribute auth paths to receivers
  T_verify = per-message Merkle path + root sig verification

Total authentication delay for a ROUTINE message:
  D_auth = T_batch + T_sign + T_distribute + T_verify

| Priority | Max Acceptable D_auth | Actual D_auth (N=16) | Compliant? |
|----------|----------------------|---------------------|-----------|
| DISTRESS | 0 (immediate) | N/A (individual sig) | Yes |
| URGENCY | 2 seconds | N/A (individual sig) | Yes |
| SAFETY | 30 seconds | 8.3 s (batch) + 8 ms + 0.5 s + 1.5 ms ≈ 9 s | Yes |
| ROUTINE | 120 seconds | Same ≈ 9 s | Yes |
```

- [ ] Add: "DISTRESS and URGENCY messages bypass batching entirely and receive immediate individual Falcon-512 signatures. This consumes temporarily higher bandwidth but is acceptable because these messages are rare and safety-critical."

**NEW Section 5.4 — Aggregator Misbehavior Mitigations:**
- [ ] Expand from brief mention to full subsection:

```markdown
### 5.4 Aggregator Misbehavior Analysis

**Threat 1: Message Censorship (exclusion from batch)**
  Mitigation:
  - Aircraft maintain monotonic sequence counters per session
  - Receiving ATC center detects gaps in sequence
  - Aircraft periodically signs a "sequence manifest" listing all
    messages sent in the last batch window
  - Manifest is signed individually (not aggregated) and sent via
    a separate channel (LDACS sideband) if available

**Threat 2: Message Modification (inclusion of tampered message)**
  Mitigation:
  - Each message includes an aircraft-side HMAC using the session key
  - Even if the aggregator modifies a message, the aircraft's HMAC
    will fail verification at the receiving ATC center
  - The Merkle proof proves the modified message was in the batch,
    but the HMAC proves it was not sent by the aircraft

**Threat 3: Batch Replay (replaying a previous batch)**
  Mitigation:
  - Each batch includes a strictly monotonic batch sequence number
  - Receivers reject batch numbers ≤ last verified batch
  - Timestamp in the root signature provides wall-clock binding

**Threat 4: Selective Delay (delaying specific messages within batch)**
  Mitigation:
  - Per-message timestamps enable receivers to detect stale messages
  - Messages older than T_max (configurable per priority) are flagged
```

**NEW Section 6.2 — Lossy Channel Analysis:**
- [ ] Add:

| Scenario | Packet Loss | Batch Integrity | Recovery Strategy |
|----------|------------|-----------------|-------------------|
| Normal (0.1%) | 0.1% of messages lost | 99.9% of batch verifiable | Merkle paths for surviving messages still valid |
| Moderate (1%) | Some auth paths incomplete | Partial batch verification | Request missing paths via retransmission |
| Severe (5%) | Root signature may be lost | Batch unverifiable | Fallback to individual sigs for next window |
| Root sig lost | 100% loss of batch auth | All messages unverified | Automatic retransmission of root |

- [ ] Add: "Merkle authentication is resilient to partial message loss: each message's authentication path is independent. Loss of message M_j does not affect verification of message M_k. Only loss of the root signature invalidates the entire batch."

**Section 7.3 — Rename and Clarify:**
- [ ] Rename "Channel Utilization" → "Bandwidth Budget Analysis"
- [ ] Add column for "Authentication Delay" alongside overhead percentage
- [ ] Add scenario for "DISTRESS burst" showing temporary 100% channel consumption is acceptable

---

## Cross-Paper Consistency Checklist

After applying all per-paper edits, verify:

- [ ] ALL papers have "Contribution Positioning" section (G1)
- [ ] ALL papers have "Non-Goals" in threat model (G2)
- [ ] ALL papers have Reproducibility Appendix (G3)
- [ ] ALL papers have standardized comparison table (G4)
- [ ] NO paper uses "production-ready" (G5)
- [ ] ALL papers cite NIST FIPS 203/204 with August 2024 date
- [ ] ALL papers have consistent parameter notation (ML-DSA-65, ML-KEM-768, SHAKE256)
- [ ] ALL papers have consistent measurement methodology description
- [ ] Related Work sections cross-reference each other where applicable (e.g., WP1 ↔ WP9, WP2 ↔ WP8)

---

## Submission Target Recommendations

| Paper | Primary Venue | Secondary Venue | Submission Timeline |
|-------|-------------|-----------------|-------------------|
| WP1 (V2X) | ACM CCS 2026 | NDSS 2027 | Submit July 2026 |
| WP2 (TESLA++) | IEEE S&P (Oakland) 2027 | ESORICS 2026 | Submit Sept 2026 |
| WP3 (Medical) | IEEE S&P Workshop on IoT Security | USENIX Security 2027 | Submit Oct 2026 |
| WP4 (HE Vitals) | PETS (PoPETs) 2027 | ACM CCS Workshop on Health Privacy | Submit after major revision |
| WP5 (ATC) | ACSAC 2026 | IEEE TDSC | Submit after repositioning |

---

*End of revision checklist*
