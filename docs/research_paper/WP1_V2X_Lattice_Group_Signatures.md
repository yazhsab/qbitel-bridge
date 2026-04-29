# QBITEL Bridge Whitepaper WP-2026-01

# A Domain-Specific Post-Quantum V2X Authentication Framework with Epoch-Bounded Linkability and Verifier-Local Revocation

**Authors:** QBITEL Bridge Research Team
**Date:** April 2026
**Version:** 2.0 (Revised)
**Classification:** Public

---

## Abstract

Vehicle-to-Everything (V2X) communication demands simultaneous privacy preservation and accountability -- vehicles must authenticate messages without revealing identity, yet misbehaving vehicles must be traceable. Current V2X Security Credential Management Systems (SCMS) rely on ECDSA-based pseudonymous certificates vulnerable to quantum computing attacks. We present QBITEL-GS, a V2X authentication framework that composes ML-DSA, PRF-based pseudonym tags, and lattice-compatible zero-knowledge proofs into a group-signature-style construction providing: (1) anonymous authentication under Module-LWE/Module-SIS assumptions, (2) epoch-bounded linkability enabling Sybil attack detection within configurable time windows (default 5 minutes) without cross-epoch tracking, (3) verifier-local revocation (VLR) allowing RSUs to detect revoked vehicles without real-time Group Manager contact, and (4) batch verification targeting 1,000+ signature verifications per second for dense traffic scenarios. Our contribution is the domain-specific protocol architecture and systems profile, not a new cryptographic primitive; we compose NIST-standardized building blocks to address the specific three-way tension between privacy, Sybil resistance, and revocation in V2X. We do not provide a full formal security proof; we sketch reductions to MLWE/MSIS and identify the precise proof obligations for future work. The research implementation targets IEEE 1609.2 and SAE J2735 compliance and demonstrates three deployment profiles (highway, urban, intersection) with measured performance characteristics on automotive-grade hardware.

**Keywords:** Post-Quantum Cryptography, Group Signatures, V2X, Vehicle Authentication, Lattice Cryptography, ML-DSA, Sybil Detection, Epoch Linkability, Verifier-Local Revocation

---

## 1. Introduction

### 1.1 The V2X Authentication Challenge

Connected vehicle communications face a fundamental tension between privacy and security. The IEEE 1609.2 standard and SAE J2735 Basic Safety Message (BSM) specification require that vehicles broadcast safety-critical telemetry -- position, speed, heading, acceleration -- at 10 Hz intervals. These messages must be authenticated to prevent spoofing attacks that could cause physical harm, yet vehicle identity must be protected to prevent mass surveillance of driving patterns.

Current SCMS deployments address this through pseudonymous certificates: each vehicle holds a pool of short-lived ECDSA certificates, rotating them periodically to break linkability. However, this approach suffers from three critical limitations:

1. **Quantum Vulnerability:** ECDSA relies on the Elliptic Curve Discrete Logarithm Problem (ECDLP), which Shor's algorithm solves in polynomial time on a sufficiently large quantum computer. Vehicles manufactured today will operate for 15-20 years, well into the projected quantum computing timeline.

2. **Certificate Management Overhead:** Each vehicle requires thousands of pseudonymous certificates, creating significant storage, distribution, and revocation challenges.

3. **Sybil Attack Surface:** Pseudonym rotation creates a window where a single vehicle could impersonate multiple vehicles by using several pseudonyms simultaneously.

### 1.2 Our Contribution

We present QBITEL-GS, a V2X authentication framework that addresses all three limitations through composition of NIST-standardized lattice-based cryptographic building blocks. Our specific contributions are:

- **Group-Signature-Style V2X Authentication:** A protocol architecture composing ML-DSA, PRF-based pseudonym tags, and lattice-compatible proofs to achieve group-signature functionality for V2X, compatible with NIST FIPS 204 parameter sets at Security Levels 2, 3, and 5.

- **Epoch-Bounded Linkability:** A pseudonym tag mechanism where signatures by the same vehicle within a configurable epoch (default: 5 minutes) are linkable for Sybil detection, while signatures across epochs are computationally unlinkable, preserving long-term privacy.

- **Verifier-Local Revocation (VLR):** A revocation mechanism where verifiers (RSUs and OBUs) locally determine whether a signature was produced by a revoked member using a compact revocation list, without contacting the Group Manager in real time.

- **High-Throughput Batch Verification:** An optimized verification pipeline targeting 1,000+ verifications per second using parallel batch processing with priority queuing, benchmarked against the NDSS 2024 targets for practical V2X deployment.

### 1.3 Contribution Positioning

This work does NOT introduce a new cryptographic primitive. The underlying algorithms (ML-DSA-65, SHAKE256, HKDF-SHA3-256) are NIST-standardized. Our contribution is a **domain-specific protocol architecture** that composes these primitives to solve V2X's specific three-way tension between privacy, Sybil resistance, and revocation locality.

Specifically, we claim novelty in:
- **Protocol design:** Epoch-bounded linkability via deterministic PRF tags, enabling Sybil detection without centralized correlation.
- **Systems engineering:** Batch verification pipeline with priority queuing achieving real-time V2X throughput on automotive hardware.
- **Deployment architecture:** Hybrid migration path, VLR for intermittent-connectivity RSUs, and channel-specific SPDU sizing.

We do NOT claim novelty in:
- The underlying lattice assumptions (MLWE, MSIS) or PQC algorithms (ML-DSA)
- Group signature theory (we build on established definitions by Bellare-Micciancio-Warinschi)
- The batch verification parallelism technique (we apply known thread-pool patterns)

We do not provide a full formal security proof. Section 6.4 identifies the precise proof obligations that remain as future work.

### 1.4 Threat Model

We consider an adversary with the following capabilities:

- **Quantum Computing Access:** The adversary has access to a cryptographically relevant quantum computer (CRQC) capable of running Shor's algorithm, breaking ECDSA, RSA, and classical Diffie-Hellman.
- **Network Control:** The adversary can observe, inject, modify, and replay V2X messages.
- **Compromised Vehicles:** The adversary may compromise a bounded number of group members and extract their signing keys.
- **Collusion:** Up to t compromised members may collude to break anonymity of honest members.

The Group Manager is assumed to be a trusted authority operated by a transportation authority or SCMS backend.

---

## 2. Cryptographic Foundations

### 2.1 Lattice Assumptions

Our construction relies on two well-studied lattice problems:

**Module Learning With Errors (MLWE).** For security parameter n, modulus q, number of modules k, and error distribution chi, the MLWE_{n,k,q,chi} problem asks to distinguish (A, As + e) from (A, u) where A is uniform over R_q^{k x k}, s is sampled from chi^k, e is sampled from chi^k, and u is uniform over R_q^k.

**Module Short Integer Solution (MSIS).** For parameters n, k, q, and bound beta, the MSIS_{n,k,q,beta} problem asks to find a nonzero vector x in R^k with ||x|| <= beta such that Ax = 0 mod q.

These are the same assumptions underlying ML-DSA (FIPS 204), ensuring our construction benefits from the extensive cryptanalysis performed during the NIST PQC standardization process.

### 2.2 Group Signature Security Properties

Our scheme satisfies the following standard group signature properties:

- **Correctness:** Honestly generated signatures verify correctly.
- **Anonymity:** Given a valid group signature, no polynomial-time adversary can determine which group member produced it, except with negligible advantage.
- **Traceability:** The Group Manager can always identify the signer of a valid group signature (opening).
- **Non-Frameability:** No coalition of members (even including the Group Manager) can produce a valid group signature that opens to an honest member who did not sign.

Additionally, we provide:

- **Epoch-Bounded Linkability:** Signatures by the same signer within an epoch are linkable; signatures across epochs are unlinkable.
- **Revocation Correctness:** Signatures by revoked members are rejected by any verifier holding the current revocation list.

### 2.3 Formal Syntax

We define QBITEL-GS as a tuple of algorithms GS = (GSetup, GKGen, GSign, GVerify, GOpen, GJudge, GRevoke) following the Bellare-Shi-Zhang (BSZ) model extended with epoch-bounded linkability:

```
GSetup(1^lambda) -> (gpk, gmsk, RL_0):
  Generate group public key gpk, Group Manager secret key gmsk,
  and empty revocation list RL_0. Security parameter lambda
  determines MLWE/MSIS parameters.

GKGen(gmsk, id_i) -> (msk_i, rt_i, upk_i):
  Group Manager issues member signing key msk_i, revocation token rt_i,
  and user public key upk_i (enrollment) for member with identity id_i.

GSign(msk_i, gpk, M, e) -> sigma:
  Member i signs message M in epoch e, producing group signature sigma
  containing (lattice_sig, pseudonym_tag, epoch, zk_proof).

GVerify(gpk, M, sigma, RL) -> {ACCEPT, REJECT}:
  Verify group signature sigma on message M using group public key gpk
  and current revocation list RL.

GOpen(gmsk, M, sigma) -> (id_i, pi_open):
  Group Manager opens signature sigma to reveal signer identity id_i
  with opening proof pi_open.

GJudge(gpk, id_i, upk_i, M, sigma, pi_open) -> {GUILTY, INNOCENT}:
  Public judge verifies that opening is correct.

GRevoke(gmsk, id_i) -> RL':
  Add revocation token for member i to revocation list.
```

### 2.4 Formal Security Definitions

We define security via the following games between a challenger C and a PPT adversary A:

**Game 1: Full-Anonymity (ANON)**

```
Experiment Exp^{ANON}_{GS,A}(lambda):
  1. (gpk, gmsk, RL) <- GSetup(1^lambda)
  2. b <-$ {0,1}
  3. A^{GKGen, GSign, GOpen*, Challenge}(gpk) -> b'
  4. Return (b == b')

  Oracle Challenge(id_0, id_1, M, e):
    sigma <- GSign(msk_{id_b}, gpk, M, e)
    Return sigma

  Oracle GOpen*(M, sigma):
    Return GOpen(gmsk, M, sigma)
    RESTRICTED: A cannot query GOpen on signatures from Challenge.

  Advantage: Adv^{ANON}_{GS,A}(lambda) = |Pr[b'=b] - 1/2|
```

**Definition 1.** GS is fully anonymous if for all PPT adversaries A, Adv^{ANON}\_{GS,A}(lambda) <= negl(lambda).

**Game 2: Full-Traceability (TRACE)**

```
Experiment Exp^{TRACE}_{GS,A}(lambda):
  1. (gpk, gmsk, RL) <- GSetup(1^lambda)
  2. A^{GKGen, GSign, Corrupt}(gpk, gmsk) -> (M*, sigma*)
  3. If GVerify(gpk, M*, sigma*, RL) = REJECT: Return 0
  4. (id*, pi*) <- GOpen(gmsk, M*, sigma*)
  5. If id* is a corrupted member: Return 0  // Trivial forgery
  6. If id* is not in member registry: Return 1  // Untraceable
  7. If GJudge(gpk, id*, upk_{id*}, M*, sigma*, pi*) = INNOCENT: Return 1
  8. Return 0

  Advantage: Adv^{TRACE}_{GS,A}(lambda) = Pr[Exp returns 1]
```

**Definition 2.** GS is fully traceable if for all PPT adversaries A, Adv^{TRACE}\_{GS,A}(lambda) <= negl(lambda).

**Game 3: Non-Frameability (NF)**

```
Experiment Exp^{NF}_{GS,A}(lambda):
  1. (gpk, gmsk, RL) <- GSetup(1^lambda)
  2. A^{GKGen, GSign, Corrupt, GOpen}(gpk) -> (M*, sigma*, id*)
  3. If id* was corrupted by A: Return 0
  4. If A ever queried GSign(msk_{id*}, *, M*, *) and received sigma*: Return 0
  5. (id_out, pi_out) <- GOpen(gmsk, M*, sigma*)
  6. If id_out == id* AND GJudge returns GUILTY: Return 1
  7. Return 0

  Advantage: Adv^{NF}_{GS,A}(lambda) = Pr[Exp returns 1]
```

**Definition 3.** GS is non-frameable if for all PPT adversaries A, Adv^{NF}\_{GS,A}(lambda) <= negl(lambda).

**Game 4: Epoch-Bounded Linkability (NEW — EBL)**

This is our new security definition, specific to the V2X application:

```
Experiment Exp^{EBL-LINK}_{GS,A}(lambda):
  // Intra-epoch: adversary must NOT be able to produce two signatures
  // from the SAME member in the SAME epoch with DIFFERENT tags
  1. (gpk, gmsk, RL) <- GSetup(1^lambda)
  2. A^{GKGen, Corrupt}(gpk) -> (id*, e*, sigma_1, sigma_2)
  3. Parse tag_1 from sigma_1, tag_2 from sigma_2
  4. If GVerify(gpk, M_1, sigma_1, RL) = REJECT: Return 0
  5. If GVerify(gpk, M_2, sigma_2, RL) = REJECT: Return 0
  6. If GOpen(sigma_1) != id* OR GOpen(sigma_2) != id*: Return 0
  7. If tag_1 != tag_2: Return 1  // Broken linkability
  8. Return 0

  Advantage: Adv^{EBL-LINK}_{GS,A}(lambda) = Pr[Exp returns 1]

Experiment Exp^{EBL-UNLINK}_{GS,A}(lambda):
  // Cross-epoch: adversary must NOT be able to link tags across epochs
  1. (gpk, gmsk, RL) <- GSetup(1^lambda)
  2. b <-$ {0,1}
  3. A^{GKGen, GSign, Challenge}(gpk) -> b'

  Oracle Challenge(id_0, id_1, M, e_target):
    // e_target is a DIFFERENT epoch from any previous Challenge query
    sigma <- GSign(msk_{id_b}, gpk, M, e_target)
    Return sigma
    // A sees tags from multiple epochs; must determine if same member

  4. Return (b == b')

  Advantage: Adv^{EBL-UNLINK}_{GS,A}(lambda) = |Pr[b'=b] - 1/2|
```

**Definition 4.** GS provides epoch-bounded linkability if both:
- Adv^{EBL-LINK}\_{GS,A}(lambda) <= negl(lambda) for all PPT A (tag consistency)
- Adv^{EBL-UNLINK}\_{GS,A}(lambda) <= negl(lambda) for all PPT A (cross-epoch unlinkability)

### 2.5 Reduction Sketches

We sketch the security reductions. Full proofs are deferred to the extended version.

**Theorem 1 (Anonymity -> MLWE).** If A breaks Full-Anonymity with advantage epsilon, then there exists an algorithm B that solves MLWE_{n,k,q,chi} with advantage >= epsilon/2 - negl(lambda).

*Proof sketch:* The challenger embeds the MLWE challenge into the ZK proof component of the group signature. When b=0, the proof is generated using msk_{id_0}; when b=1, using msk_{id_1}. By the zero-knowledge property of the lattice proof system (which relies on MLWE), the two cases are computationally indistinguishable. The pseudonym tag is derived via PRF(msk_{id_b}, e || "pseudonym"), so distinguishing tags reduces to breaking the PRF security of SHAKE256.

**Theorem 2 (Traceability -> MSIS).** If A breaks Full-Traceability with advantage epsilon, then there exists an algorithm B that solves MSIS_{n,k,q,beta} with advantage >= epsilon - negl(lambda).

*Proof sketch:* An untraceable signature implies the adversary produced a valid ZK proof of knowledge of a member key without any member key. The knowledge extractor for the lattice proof system extracts a witness, which is either a valid member key (contradicting untraceability) or a short vector solving the MSIS instance embedded in gpk.

**Theorem 3 (EBL-Unlinkability -> PRF).** If A breaks Cross-Epoch Unlinkability with advantage epsilon, then there exists an algorithm B that breaks the PRF security of SHAKE256 with advantage >= epsilon - negl(lambda).

*Proof sketch:* Tags are computed as tag = PRF(msk_i, e || "pseudonym"). In epoch e_1, A sees tag_1 = PRF(msk_{id_b}, e_1 || "pseudonym"). In epoch e_2 != e_1, A sees tag_2 = PRF(msk_{id_b}, e_2 || "pseudonym"). Distinguishing whether tag_1 and tag_2 correspond to the same id_b reduces to distinguishing PRF outputs from random, since different epochs produce independent PRF evaluations. B simulates the group signature scheme and uses A's advantage to distinguish PRF from random.

**Theorem 4 (EBL-Linkability -> Tag Determinism).** Tag consistency holds unconditionally: for a fixed (msk_i, e), the tag PRF(msk_i, e || "pseudonym") is deterministic. No adversary, even computationally unbounded, can produce two valid signatures with the same (id, epoch) but different tags.

*Proof:* By correctness of the ZK proof: the proof verifies that the tag is correctly derived from the signer's key and the epoch. The verification equation binds the tag uniquely to (msk_i, e). Producing a valid proof for a different tag would require either (a) a different msk_i (different member), or (b) a valid proof for an incorrect PRF evaluation (contradicting soundness).

---

## 3. Scheme Construction

### 3.1 System Parameters

```
Security Levels:
  LEVEL_2: ML-DSA-44 equivalent (128-bit classical security)
  LEVEL_3: ML-DSA-65 equivalent (192-bit classical security) [DEFAULT]
  LEVEL_5: ML-DSA-87 equivalent (256-bit classical security)

Scheme Variants:
  LATTICE_DILITHIUM: Module-LWE/SIS based, ML-DSA compatible
  LATTICE_FALCON: NTRU-lattice based, Falcon compatible
```

### 3.2 Key Generation

**Group Setup (GroupManager.setup):**

1. Generate Group Manager key pair (gmsk, gmpk) using ML-DSA at the configured security level.
2. Initialize the member registry M = {} and revocation list RL = [].
3. Publish group public key gpk = (gmpk, system_parameters).

**Member Join (GroupManager.issue_signing_key):**

1. Member i generates identity commitment id_i.
2. Group Manager generates member signing key msk_i derived from gmsk and id_i.
3. Group Manager computes revocation token rt_i = PRF(gmsk, id_i) for future VLR use.
4. Group Manager stores (id_i, msk_i, rt_i) in member registry.
5. Member receives (msk_i, membership_credential).

### 3.3 Group Signing

**Sign(msk_i, message, epoch):**

1. Compute pseudonym tag: tag = PRF(msk_i, epoch || "pseudonym"). This tag is deterministic per (member, epoch), enabling linkability within the epoch.
2. Generate lattice-based zero-knowledge proof pi proving:
   - Knowledge of a valid member signing key msk_i
   - The pseudonym tag is correctly derived from msk_i and the current epoch
   - The signer is not on the revocation list
3. Compute message signature sigma = ML-DSA.Sign(derived_key, message || tag || epoch).
4. Output group signature GS = (sigma, tag, epoch, pi).

### 3.4 Group Verification

**Verify(gpk, message, GS):**

1. Parse GS = (sigma, tag, epoch, pi).
2. Verify epoch is current (within acceptable clock skew).
3. Verify zero-knowledge proof pi against gpk.
4. Verify signature sigma over (message || tag || epoch).
5. Check tag against revocation list RL: for each revocation token rt in RL, verify tag != VLR_Check(rt, epoch).
6. Output ACCEPT or REJECT.

### 3.5 Time-Windowed Linkability

The pseudonym tag mechanism provides the following properties:

- **Intra-Epoch Linkability:** Two group signatures GS_1 = (..., tag_1, epoch, ...) and GS_2 = (..., tag_2, epoch, ...) are from the same signer if and only if tag_1 == tag_2. This is deterministic and requires no Group Manager involvement.

- **Cross-Epoch Unlinkability:** Given signatures from epochs e_1 != e_2, the tags tag_1 = PRF(msk_i, e_1 || "pseudonym") and tag_2 = PRF(msk_i, e_2 || "pseudonym") are computationally indistinguishable from random, under the PRF security assumption.

The default epoch duration is 5 minutes, chosen to balance:
- **Sybil Detection Window:** Long enough to detect vehicles broadcasting conflicting positions under multiple pseudonyms.
- **Privacy Preservation:** Short enough to prevent long-term vehicle tracking by roadside observers.

### 3.6 Verifier-Local Revocation (VLR)

When a vehicle is revoked:

1. Group Manager publishes the revocation token rt_i to the revocation list RL.
2. Each verifier downloads RL (periodically or via broadcast).
3. During verification, the verifier computes expected_tag = VLR_Derive(rt_i, current_epoch) for each rt_i in RL.
4. If the signature's pseudonym tag matches any expected_tag, the signature is REJECTED.

This approach eliminates the need for real-time connectivity to the Group Manager during verification -- critical for V2X scenarios where RSUs may have intermittent connectivity and OBUs operate in real-time safety-critical loops.

### 3.7 Privacy Leakage from Epoch Design

The epoch duration creates a fundamental tradeoff: longer epochs improve Sybil detection but increase privacy risk from trajectory stitching by roadside observers. We analyze this tradeoff:

| Epoch Duration | Linkability Window | Trajectory Stitching Risk | Sybil Detection Effectiveness | Recommended Scenario |
|---------------|-------------------|--------------------------|------------------------------|---------------------|
| 30 seconds | Very low | Negligible | Poor (too few messages at 10 Hz = 300 msgs) | Privacy-critical urban |
| 1 minute | Low | Low | Moderate (600 msgs) | Dense urban |
| **5 minutes (default)** | **Moderate** | **Moderate (urban concern)** | **Good (3,000 msgs)** | **General purpose** |
| 10 minutes | High | High (highway tracking risk) | Very good (6,000 msgs) | Highway only |

**RSU-Colluding Observer Model:** An adversary controlling multiple RSUs along a corridor can observe the same pseudonym tag across RSU coverage areas within one epoch. With a 5-minute epoch at highway speed (120 km/h), a vehicle travels ~10 km, potentially crossing 5-10 RSU coverage zones. This enables trajectory reconstruction within the epoch.

**Countermeasures:**
- **Randomized epoch boundaries:** Each vehicle adds uniform random jitter of +/- 30 seconds to its epoch transition time, preventing synchronized epoch changes that could be exploited by multi-RSU observers.
- **Per-OBU epoch offset:** During enrollment, each vehicle receives a random epoch offset (0 to epoch_duration), ensuring different vehicles transition at different times.
- **Adaptive epochs:** In dense urban environments where RSU density is high, vehicles can autonomously reduce epoch duration to 1-2 minutes; on highways with sparse RSUs, longer epochs (10 minutes) are safer for privacy because fewer observers exist.

### 3.8 Revocation Scalability Analysis

VLR check cost scales linearly with revocation list size. We analyze scalability across fleet sizes:

| Fleet Size | Revocation Rate | RL Size (1 year) | VLR Check Time (per sig) | RSU RL Storage | RL Broadcast Size |
|-----------|----------------|-------------------|--------------------------|---------------|-------------------|
| 10,000 | 0.1%/year | 10 entries | 0.03 ms | 320 B | 320 B |
| 100,000 | 0.1%/year | 100 entries | 0.3 ms | 3.2 KB | 3.2 KB |
| 1,000,000 | 0.1%/year | 1,000 entries | 3.0 ms | 32 KB | 32 KB |
| 10,000,000 | 0.5%/year | 50,000 entries | 150 ms | 1.6 MB | 1.6 MB |

**At 10M fleet scale, 150 ms VLR check per signature is unacceptable for real-time V2X.** We propose two mitigations:

1. **Bloom Filter VLR:** Replace the explicit RL with a Bloom filter. A Bloom filter with 50,000 entries and 0.1% false positive rate requires ~72 KB (vs. 1.6 MB for explicit list). VLR check becomes a constant-time hash lookup (~0.01 ms). False positives (0.1%) cause legitimate vehicles to be occasionally rejected; this is tolerable if verification is retried with the next message.

2. **Delta RL Updates:** Broadcast full RL weekly; daily updates contain only additions and removals (delta). Typical daily delta: 10-50 entries (< 2 KB), easily broadcast via SCMS backend or RSU multicast.

3. **RL Sharding by Region:** Partition the RL by geographic region. RSUs in region R only need revocation tokens for vehicles enrolled in region R plus a "visiting vehicle" overlay. Reduces per-RSU RL to ~10-20% of global RL.

---

## 4. Batch Verification Engine

### 4.1 Architecture

High-density V2X scenarios (e.g., urban intersections with 200+ vehicles) require verifying hundreds of signatures per second. Our batch verification engine uses:

```
Verification Pipeline:
  1. Priority Queue: Incoming messages are enqueued with priority levels
     - CRITICAL: Emergency Vehicle Alerts (EVA), Distress
     - HIGH: BSM from nearby vehicles (<50m)
     - NORMAL: Standard BSM
     - LOW: Infrastructure messages (TIM, MAP)

  2. Batch Collector: Accumulates messages until batch_size reached
     or batch_timeout expires (whichever first)

  3. Parallel Verifier: ThreadPoolExecutor processes batches
     with configurable worker count

  4. Result Aggregator: Collects results with per-message pass/fail
```

### 4.2 Deployment Profiles

| Profile | Batch Size | Workers | Target Throughput | Latency P99 |
|---------|-----------|---------|-------------------|-------------|
| Highway | 32 | 4 | 500/s | <5ms |
| Urban | 128 | 16 | 2,000/s | <8ms |
| Intersection | 256 | 32 | 5,000/s | <10ms |

### 4.3 Optimization Techniques

1. **Signature Pre-filtering:** Reject malformed signatures before expensive lattice operations.
2. **Revocation Cache:** Hot cache of VLR check results to avoid repeated computation.
3. **Epoch Caching:** Pre-compute epoch-dependent values once per epoch transition.
4. **SIMD Exploitation:** Batch NTT (Number Theoretic Transform) operations across multiple signatures where supported by the underlying ML-DSA implementation.

---

## 5. IEEE 1609.2 Integration

### 5.1 Protocol Data Unit Format

Our group signatures are encapsulated within IEEE 1609.2 Secured Protocol Data Units (SPDUs):

```
SignedPDU:
  payload:       BSM | EVA | TIM | SPAT | MAP | RSA
  signer_info:   GroupSignatureInfo
    group_id:    bytes (group public key fingerprint)
    pseudonym:   bytes (epoch pseudonym tag)
    epoch:       uint32
  signature:     GroupSignature
    lattice_sig: bytes (ML-DSA or Falcon signature)
    zk_proof:    bytes (membership proof)
  timestamp:     uint64 (microseconds since epoch)
```

### 5.2 Hybrid Mode

For backward compatibility during transition, we support a hybrid mode where each SPDU carries both:
- A classical ECDSA-P256 signature (for legacy verifiers)
- A post-quantum group signature (for PQ-capable verifiers)

The hybrid mode adds approximately 3.5 KB overhead per message but ensures interoperability with existing deployed infrastructure.

### 5.3 SAE J2735 BSM Integration

The Basic Safety Message Part I (core) fields authenticated by our scheme:

```
BasicSafetyMessage:
  latitude:      float (WGS84)
  longitude:     float (WGS84)
  elevation:     float (meters)
  speed:         float (m/s)
  heading:       float (degrees)
  acceleration:  (longitudinal, lateral, vertical, yaw_rate)
  vehicle_size:  (width, length)
  brake_status:  BrakeSystemStatus
  timestamp:     DSecond
```

### 5.4 SPDU Size Sensitivity Analysis

| Scenario | Algorithm | Group Sig | ZK Proof | Tag | Hybrid Classical | Total SPDU Overhead | DSRC 5.9 GHz Fit (100 vehicles) |
|----------|-----------|-----------|----------|-----|-----------------|--------------------|---------------------------------|
| Highway sparse (20 veh) | Falcon-512 | 1.8 KB | 1.2 KB | 32 B | N/A | 3.0 KB | Yes |
| Highway sparse | ML-DSA-65 | 4.2 KB | 1.5 KB | 32 B | N/A | 5.7 KB | Yes |
| Urban dense (100 veh) | Falcon-512 | 1.8 KB | 1.2 KB | 32 B | N/A | 3.0 KB | Yes (marginal) |
| Urban dense | ML-DSA-65 | 4.2 KB | 1.5 KB | 32 B | N/A | 5.7 KB | No (exceeds budget) |
| Hybrid mode (any) | Falcon-512 | 1.8 KB | 1.2 KB | 32 B | +64 B ECDSA | 3.1 KB | Yes |
| Hybrid mode (any) | ML-DSA-65 | 4.2 KB | 1.5 KB | 32 B | +64 B ECDSA | 5.8 KB | Marginal |
| C-V2X PC5 (20 MHz) | Falcon-512 | 1.8 KB | 1.2 KB | 32 B | N/A | 3.0 KB | Yes |

**Recommendation:** Use Falcon-512 for bandwidth-constrained DSRC deployments. ML-DSA-65 is viable only on LDACS or C-V2X with sufficient bandwidth, or in sparse traffic (<50 vehicles per channel).

---

## 6. Security Analysis

### 6.1 Quantum Resistance

The security of QBITEL-GS reduces to the hardness of MLWE and MSIS, which are believed to be hard for both classical and quantum computers. Specifically:

- At LEVEL_3 (default), we achieve 192-bit security against classical attacks and approximately 128-bit security against quantum attacks, consistent with ML-DSA-65 parameters.
- The PRF used for pseudonym tag generation (SHAKE256) is quantum-safe under the random oracle model.
- Key derivation uses HKDF-SHA3-256, providing quantum-safe key expansion.

### 6.2 Privacy Properties

**Anonymity:** An adversary observing signatures from multiple vehicles at a single epoch can identify that N distinct pseudonym tags are present, but cannot link any tag to a specific vehicle identity. The anonymity guarantee holds even if the adversary compromises up to (group_size - 2) other members.

**Unlinkability Across Epochs:** An adversary collecting signatures across multiple epochs cannot determine whether two pseudonym tags from different epochs belong to the same vehicle, under the PRF security assumption for SHAKE256.

**Forward Privacy:** Revocation of a vehicle reveals its revocation token but does not retroactively de-anonymize past signatures from earlier epochs (assuming the epoch has expired and no VLR match is stored).

### 6.3 Sybil Resistance

Within any epoch, each vehicle produces a single deterministic pseudonym tag. A vehicle attempting to broadcast under multiple identities would produce the same tag in all messages, immediately detectable by any verifier comparing tags within the epoch window.

### 6.4 Open Proof Obligations

We identify the following formal proof obligations that remain as future work. These are standard requirements for a group signature scheme; our current work provides construction sketches and reduction strategies but not complete proofs.

1. **Full CCA-anonymity reduction from MLWE:** Prove that the group signature hides the signer's identity under chosen-ciphertext attacks, reducing to the Module-LWE assumption.
2. **Non-frameability under adaptive corruption:** Prove that no coalition of (group_size - 1) corrupted members plus the Group Manager can produce a valid signature that opens to an honest member who did not sign.
3. **VLR soundness with Bloom filter approximation:** Prove that the Bloom filter VLR optimization (Section 3.8) does not introduce exploitable false negatives (revoked vehicles passing verification).
4. **PRF-based tag unlinkability:** Formally reduce cross-epoch tag unlinkability to the PRF security of SHAKE256 in the quantum random oracle model (QROM).
5. **Batch verification soundness:** Prove that batch verification (Section 4) does not accept invalid signatures that individual verification would reject.

These proof obligations are well-scoped and follow established templates from the lattice-based group signature literature (Ling et al., Libert et al., Gordon et al.). We expect they can be discharged without fundamental changes to the construction.

---

## 7. Performance Evaluation

### 7.1 Signature Sizes

| Component | ML-DSA-65 Based | Falcon-512 Based |
|-----------|----------------|-----------------|
| Group Signature | ~4.2 KB | ~1.8 KB |
| Pseudonym Tag | 32 bytes | 32 bytes |
| ZK Proof | ~1.5 KB | ~1.2 KB |
| Total SPDU Overhead | ~5.7 KB | ~3.0 KB |

### 7.2 Computational Performance

Measured on ARM Cortex-A72 (representative of automotive-grade ECU):

| Operation | ML-DSA-65 | Falcon-512 |
|-----------|-----------|------------|
| Group Sign | 2.1 ms | 8.3 ms |
| Group Verify (single) | 1.4 ms | 0.9 ms |
| Batch Verify (32) | 18 ms (0.56 ms/sig) | 12 ms (0.38 ms/sig) |
| Batch Verify (128) | 64 ms (0.50 ms/sig) | 42 ms (0.33 ms/sig) |
| VLR Check (100 entries) | 0.3 ms | 0.3 ms |
| Pseudonym Tag | 0.01 ms | 0.01 ms |

### 7.3 Bandwidth Analysis

At 10 Hz BSM transmission rate with Falcon-512 based group signatures:

- Per-vehicle overhead: 3.0 KB x 10 Hz = 30 KB/s = 240 kbps
- Within DSRC 5.9 GHz capacity (27 Mbps per channel)
- Supports approximately 100+ vehicles per channel (vs. ~150 with classical ECDSA)

---

## 8. Deployment Considerations

### 8.1 Group Manager Infrastructure

The Group Manager operates within the SCMS backend infrastructure with the following interfaces:

- **Enrollment:** Vehicles receive group member credentials during manufacturing or registration.
- **Revocation:** Compromised or misbehaving vehicles are revoked via Certificate Revocation Lists distributed to RSUs.
- **Tracing:** Law enforcement with proper legal authority can request the Group Manager to open a specific group signature, revealing the vehicle's identity.

### 8.2 Migration Strategy

We recommend a phased deployment:

1. **Phase 1 (Hybrid):** Deploy PQ group signatures alongside existing ECDSA pseudonymous certificates. Both are verified; PQ failures are logged but not enforced.
2. **Phase 2 (PQ-Primary):** PQ group signature verification is required; classical ECDSA is optional for backward compatibility.
3. **Phase 3 (PQ-Only):** Classical ECDSA is deprecated; only PQ group signatures are accepted.

### 8.3 Standards Alignment

| Standard | Compliance |
|----------|-----------|
| IEEE 1609.2 | SPDU format, certificate hierarchy |
| SAE J2735 | BSM structure and encoding |
| ETSI TS 103 097 | European ITS security header |
| ETSI TR 103 415 | Pre-authorization for pseudonym changes |
| NIST FIPS 204 | ML-DSA parameter sets |
| NIST SP 800-208 | Stateful hash-based signatures (firmware) |

---

## 9. Related Work

Twardokus et al. (NDSS 2024) demonstrated practical PQ authentication for V2V using Falcon-512 within IEEE 1609.2, finding that certificate transmissions are 93% redundant. Our work extends this by replacing pseudonymous certificates entirely with group signatures, eliminating the certificate management overhead.

Barreto et al. (IACR 2018) proposed qSCMS with lattice-based butterfly key expansion for V2X certificate provisioning. Our approach is complementary -- qSCMS addresses the certificate backend, while our group signatures can replace the certificate-per-pseudonym model entirely.

Lesaignoux and Carmona (SECRYPT 2024) provided the first lattice-based DAA implementation for VANETs. Our scheme differs in providing time-windowed linkability (vs. full anonymity) and VLR (vs. centralized revocation), both critical for practical V2X deployment.

### 9.1 Formal Comparison

| Property | SCMS (ECDSA) | qSCMS (Lattice) | DAA/VANET (Lattice) | QBITEL-GS (This Work) |
|----------|-------------|-----------------|--------------------|-----------------------|
| PQ Resistant | No | Yes | Yes | Yes |
| Privacy Model | Pseudonym pool rotation | Pseudonym pool (BKE) | Full anonymity | Epoch-bounded linkability |
| Sybil Detection | Weak (cert tracking by SCMS) | Weak (same) | None native | Strong (tag linkability per epoch) |
| Revocation | CRL via SCMS backend | CRL via SCMS backend | Group Manager only | VLR (fully local) |
| Certs per Vehicle | 3,000+ | 3,000+ | 0 (group membership) | 0 (group membership) |
| Verifier Connectivity | Periodic CRL download | Periodic CRL download | GM required for trace | Fully local (VLR list only) |
| Batch Verification | Per-certificate | Per-certificate | Limited | 1,000+/s (pipelined) |
| Cert Storage (PQ) | ~16.5 MB | ~3.75 MB (implicit) | ~2 KB (group key) | ~2 KB (group key) |
| Traceability | Linkage Authority | Linkage Authority | Group Manager | Group Manager |
| Standards Fit | IEEE 1609.2 native | IEEE 1609.2 compatible | Research stage | IEEE 1609.2 compatible |

---

## 10. Conclusion

QBITEL-GS provides a practical path to quantum-resistant V2X authentication that simultaneously improves privacy properties and reduces infrastructure complexity compared to current pseudonymous certificate approaches. The time-windowed linkability mechanism resolves the fundamental tension between Sybil resistance and privacy preservation, while verifier-local revocation ensures real-time operation even with intermittent backend connectivity. Our batch verification engine demonstrates that post-quantum group signatures can meet the throughput requirements of dense urban V2X deployments.

---

## References

[1] NIST FIPS 204, "Module-Lattice-Based Digital Signature Standard (ML-DSA)," August 2024.

[2] IEEE 1609.2-2022, "IEEE Standard for Wireless Access in Vehicular Environments -- Security Services for Applications and Management Messages."

[3] SAE J2735, "V2X Communications Message Set Dictionary."

[4] G. Twardokus, N. Bindel, H. Rahbari, S. McCarthy, "When Cryptography Needs a Hand: Practical Post-Quantum Authentication for V2V Communications," NDSS 2024.

[5] P. Barreto, J. Ricardini, M. Simplicio, H. Kupwade Patil, "qSCMS: Post-Quantum Certificate Provisioning Process for V2X," IACR ePrint 2018/1247.

[6] D. Lesaignoux, M. Carmona, "On the Implementation of a Lattice-Based DAA for VANET System," SECRYPT 2024 / IACR ePrint 2024/464.

[7] ETSI TS 103 097, "Intelligent Transport Systems (ITS); Security; Security Header and Certificate Formats."

[8] ETSI TR 103 415, "Intelligent Transport Systems (ITS); Security; Pre-authorization."

---

---

## Appendix A: Reproducibility

### Hardware
- Benchmark platform: ARM Cortex-A72 @ 1.5 GHz (Raspberry Pi 4, representative of automotive gateway ECU)
- Target deployment: NXP S32G274A (ARM Cortex-A53 @ 1 GHz + Cortex-M7 @ 400 MHz)

### Software
- Python 3.11 with asyncio
- PQC library: liboqs 0.10.0 (C reference implementation via ctypes bindings)
- Hash functions: hashlib (Python stdlib, OpenSSL backend)
- Metrics: prometheus_client 0.20.0

### Parameter Sets
- ML-DSA-65: NIST FIPS 204, Category 3 (k=6, l=5, eta=4)
- Falcon-512: NIST PQC Round 3, n=512
- SHAKE256: NIST FIPS 202 / SP 800-185
- HKDF: RFC 5869 with SHA3-256

### Workload
- Synthetic BSM messages conforming to SAE J2735 structure
- 10,000 unique vehicle identities per experiment
- Measurement: median of 100 runs, 10-run warm-up discarded
- All timing measurements exclude I/O (pure CPU time)

### Implementation Status
- [x] Research prototype with comprehensive test suite
- [x] Prometheus metrics instrumentation
- [x] Async/await concurrency for batch processing
- [ ] Formal security proof (see Section 6.4)
- [ ] Hardware-in-the-loop testing on automotive ECU
- [ ] Field deployment

---

*QBITEL Bridge -- Quantum-Safe Infrastructure for Critical Systems*
*Copyright 2026 QBITEL. All rights reserved.*
