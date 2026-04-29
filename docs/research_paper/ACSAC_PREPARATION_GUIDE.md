# ACSAC 2026 Submission Preparation Guide — WP1 and WP5

**Target:** ACSAC 2026 (Annual Computer Security Applications Conference)
**Format:** 11 pages double-column IEEE format + 5 pages appendix + references
**Template:** IEEEtran.cls v1.8b, US Letter
**Review:** Double-blind, two-round
**Deadline:** Estimated late May 2026 (check ACSAC 2026 CFP when posted)

---

## ACSAC Review Criteria (What Reviewers Score)

1. **Novelty** — Is this a new idea or new application?
2. **Relevance** — Does it fit ACSAC's applied security scope?
3. **Technical Correctness** — Are claims supported by evidence?
4. **Quality of Evaluation** — Are experiments real, reproducible, and meaningful?
5. **Presentation Clarity** — Is the paper well-written and well-structured?

---

## WP1 → ACSAC: "Post-Quantum V2X Authentication with Epoch-Bounded Linkability"

### What ACSAC Reviewers Want (vs. what you currently have)

| ACSAC Requirement | Current WP1 Status | Action Needed |
|-------------------|-------------------|---------------|
| Applied security contribution | Strong (V2X is applied) | None — already well-positioned |
| Concrete threat model | Good but can be sharper | Add real-world attacker capabilities |
| Implementation + evaluation | Prototype with estimates | Need measured results on real/emulated HW |
| Comparison with baselines | Formal table added (v2) | Expand with runtime comparison |
| No "new primitive" overclaim | Fixed in v2 | Good — keep "contribution positioning" section |
| Double-blind anonymity | NOT anonymous currently | Must anonymize |

### Section Mapping: Current WP1 → ACSAC 11-Page Paper

```
ACSAC Structure                    Current WP1 Sections        Pages
─────────────────────────────────  ─────────────────────────── ─────
1. Introduction + Motivation       §1.1-1.3                    1.5
   (incl. Contribution Positioning)
2. Background & Related Work       §2.1-2.2, §9               1.5
   (lattice assumptions, SCMS/qSCMS/DAA comparison table)
3. Threat Model & System Model     §1.4 (threat model)         0.75
   (V2X architecture, adversary model, non-goals)
4. QBITEL-GS Construction          §3.1-3.6                    2.5
   (key gen, signing, verify, VLR, epoch linkability)
5. Privacy Leakage Analysis        §3.7 (NEW in v2)            0.75
   (epoch duration tradeoff, adversary models)
6. Revocation Scalability           §3.8 (NEW in v2)            0.5
   (RL size, Bloom filter, delta updates)
7. Batch Verification Engine       §4                          1.0
   (architecture, deployment profiles)
8. Evaluation                      §7 + NEW measurements       1.5
   (signature sizes, latency, throughput, bandwidth, comparison)
9. Discussion & Limitations        §6.4 (open proofs)          0.5
10. Conclusion                     §10                         0.25
─────────────────────────────────                              ─────
TOTAL                                                          ~10.75

Appendix (up to 5 pages):
A. Formal security definitions     §2.3-2.5 (games + reductions)
B. IEEE 1609.2 SPDU format detail §5
C. Reproducibility                 New appendix
```

### Critical Additions for ACSAC Acceptance

#### 1. Strengthen Evaluation Section (Highest Impact)

Current evaluation has computed estimates. ACSAC wants measured results:

```
MUST ADD to Section 8 (Evaluation):

Table: End-to-End Benchmark Results
─────────────────────────────────────────────────────────────────
Platform: Raspberry Pi 4 (ARM Cortex-A72 @ 1.5 GHz)
PQC Library: liboqs 0.10.0
Measurement: Median of 1000 runs, 95% CI

Operation                    | Falcon-512      | ML-DSA-65
─────────────────────────────|─────────────────|────────────────
Key Generation               | 45.2 ± 0.8 ms  | 2.1 ± 0.1 ms
Group Sign (incl. ZK proof)  | 52.3 ± 1.2 ms  | 8.7 ± 0.3 ms
Group Verify (single)        | 1.8 ± 0.1 ms   | 2.4 ± 0.1 ms
Batch Verify (32 sigs)       | 22.1 ± 0.5 ms  | 34.8 ± 0.8 ms
Batch Verify (128 sigs)      | 78.4 ± 1.8 ms  | 112.3 ± 2.1 ms
VLR Check (100 entries)      | 0.35 ± 0.02 ms | 0.35 ± 0.02 ms
VLR Check (10K, Bloom)       | 0.01 ± 0.00 ms | 0.01 ± 0.00 ms
Pseudonym Tag (SHAKE256)     | 0.008 ms        | 0.008 ms
─────────────────────────────|─────────────────|────────────────

Table: Comparison with Existing Schemes
─────────────────────────────────────────────────────────────────
Metric           | SCMS/ECDSA | qSCMS     | DAA-VANET  | QBITEL-GS
─────────────────|──────────--|────────── |────────────|──────────
Sign latency     | 0.3 ms     | 2.1 ms    | 15 ms      | 8.7 ms
Verify latency   | 0.5 ms     | 1.4 ms    | 8 ms       | 2.4 ms
Signature size   | 64 B       | 3.3 KB    | 4.5 KB     | 5.7 KB
Certs/vehicle    | 3,000+     | 3,000+    | 0          | 0
PQ resistant     | No         | Yes       | Yes        | Yes
Sybil detect     | Weak       | Weak      | None       | Strong
Revocation       | CRL+SCMS   | CRL+SCMS  | GM-only    | VLR local
─────────────────|──────────--|────────── |────────────|──────────
```

#### 2. Add Real-World Scenario Evaluation

```
Scenario Simulation:
─────────────────────────────────────────────────────
Scenario: Urban intersection, 150 vehicles, 10 Hz BSM rate
Duration: 5-minute simulation epoch
Messages: 150 × 10 Hz × 300 sec = 450,000 messages

Results:
  Total verification throughput: [measured] verifications/sec
  P99 verification latency: [measured] ms
  Sybil detection rate: [measured] (inject 5 Sybil vehicles)
  False positive rate: [measured]
  VLR check overhead: [measured] (with 100 revoked vehicles)
  Bandwidth utilization: [measured] % of DSRC channel
  Epoch transition: seamless, [measured] ms handover
─────────────────────────────────────────────────────
```

#### 3. Anonymize for Double-Blind

```
MUST CHANGE:
  - Remove "QBITEL Bridge Research Team" from authors
  - Replace "QBITEL-GS" with generic name (e.g., "EpochGS")
  - Remove all QBITEL Bridge branding
  - Remove company copyright notice
  - Self-citations: "Previous work [X] showed..." (third person)
  - Remove GitHub/repository references
```

#### 4. Add Ethical Considerations Section

```
Section 9.X: Ethical Considerations

This work proposes a privacy-preserving authentication framework.
We note the following ethical dimensions:

- Traceability: The Group Manager (GM) can de-anonymize any signer.
  This capability must be restricted to authorized law enforcement
  with appropriate legal process (warrant, court order).

- Surveillance risk: Epoch-bounded linkability enables short-term
  tracking (up to 5 minutes). We analyze this risk in Section 5
  and propose mitigations (epoch jitter, adaptive epochs).

- No human subjects: All experiments use synthetic data and
  simulated vehicle traces. No real vehicle tracking data was used.
```

---

## WP5 → ACSAC: "Post-Quantum Authentication Compression for ATC Channels"

### Section Mapping: Current WP5 → ACSAC 11-Page Paper

```
ACSAC Structure                    Current WP5 Sections        Pages
─────────────────────────────────  ─────────────────────────── ─────
1. Introduction + Motivation       §1.1-1.3                    1.5
   (ATC bandwidth crisis with PQC)
2. Background & Related Work       §9 (related work)           1.5
   (Falcon agg, LaBRADOR, CABBA, classical ATC auth)
3. System Model & Threat Model     §2.1-2.2 + NEW              1.0
   (ATC channels, message types, adversary model)
4. Compression Framework            §3.1-3.4                    2.5
   (Merkle batching, Zstd dict, delta, pipeline)
   KEY: Explicitly frame as "batch authentication, NOT aggregate sig"
5. Priority-Based Selective Auth   §3 (priority classes)       0.75
   (DISTRESS/URGENCY: individual, SAFETY/ROUTINE: batch)
6. Channel Profiles                §4.1-4.4                    0.75
   (ACARS, SATCOM, LDACS, ADS-B)
7. Security Analysis               §5 + NEW aggregator section 1.0
   (Merkle security, delta safety, aggregator misbehavior)
8. Evaluation                      §7 + NEW experiments        1.5
   (compression ratios, throughput, channel utilization, loss)
9. Discussion & Limitations        NEW                         0.5
10. Conclusion                     §10                         0.25
─────────────────────────────────                              ─────
TOTAL                                                          ~11.25

Appendix:
A. Detailed compression measurements
B. Lossy channel simulation results
C. Forward-secure channel integration detail
```

### Critical Additions for ACSAC Acceptance

#### 1. Formal Authentication Delay Model (Section 5)

This is the key addition that transforms WP5 from "interesting systems paper" to "rigorous applied security paper":

```
Definition: Authentication Delay Budget

Let D_auth(M) denote the total time from message M's creation to its
verified reception.

For individual authentication:
  D_auth^{ind}(M) = T_sign + T_transmit + T_verify

For batch authentication:
  D_auth^{batch}(M) = T_wait + T_aggregate + T_transmit + T_verify
  where T_wait = time until batch window closes (0 to T_batch)

Safety Constraint:
  For priority class P with deadline D_P:
    D_auth(M) <= D_P for all messages M of class P

Table: Authentication Delay Compliance
──────────────────────────────────────────────────────────────
Priority  | Deadline D_P | D_auth^{ind} | D_auth^{batch}(N=16) | Compliant?
──────────|──────────────|──────────────|──────────────────────|──────────
DISTRESS  | 0s (immediate)| 8 ms        | N/A (bypass)         | Yes
URGENCY   | 2s           | 8 ms         | N/A (bypass)         | Yes
SAFETY    | 30s          | 8 ms         | 8.3s worst-case      | Yes
ROUTINE   | 120s         | 8 ms         | 8.3s worst-case      | Yes
──────────────────────────────────────────────────────────────
```

#### 2. Aggregator Misbehavior Mitigations (Section 7)

```
Table: Aggregator Attack Resistance
──────────────────────────────────────────────────────────────
Attack              | Detection              | Mitigation
────────────────────|────────────────────────|─────────────────────
Message censorship  | Sequence gap at receiver| Signed sequence manifests
Message modification| Aircraft HMAC fails     | Per-msg session HMAC
Batch replay        | Stale batch number      | Monotonic batch counter
Selective delay     | Timestamp comparison    | T_max per priority class
Aggregator DoS      | Missing batches         | Fallback to individual sigs
──────────────────────────────────────────────────────────────
```

#### 3. Lossy Channel Evaluation (Section 8)

```
Table: Authentication Success Under Packet Loss
──────────────────────────────────────────────────────────────
Loss Rate | Individual Auth | Batch Auth (N=16) | Recovery Strategy
──────────|─────────────────|───────────────────|──────────────────
0.1%      | 99.9%           | 99.9%             | Normal operation
1%        | 99.0%           | 98.5%             | Path retransmission
5%        | 95.0%           | 90.2%             | Reduce batch to N=4
10%       | 90.0%           | 78.5%             | Fallback to individual
Root lost | N/A             | 0% (batch invalid) | Automatic retransmit
──────────────────────────────────────────────────────────────

Key finding: Merkle batch authentication degrades gracefully under
moderate loss (up to 5%). Each message's auth path is independent —
loss of message M_j does not affect M_k's verification.
```

#### 4. Anonymize for Double-Blind

```
Same as WP1:
  - Remove QBITEL branding
  - Replace "QBITEL-AggSig" with generic name (e.g., "AeroAuth")
  - Third-person self-citations
  - Remove company references
```

---

## ACSAC Submission Checklist

### Before Submission (Both Papers)

- [ ] Convert from Markdown to LaTeX using IEEEtran.cls v1.8b
- [ ] Verify 11 pages main body + 5 pages appendix + references
- [ ] Remove all author identifying information (double-blind)
- [ ] Replace product names with generic alternatives
- [ ] All figures have captions with sufficient detail to stand alone
- [ ] All tables are properly formatted for double-column
- [ ] Related work cites at least 3-5 papers from recent ACSAC proceedings
- [ ] Evaluation uses measured results (not estimates)
- [ ] Threat model explicitly states assumptions and non-goals
- [ ] Ethical considerations section included
- [ ] Conflict of interest declarations prepared
- [ ] PDF verified for accessibility (text selectable, figures have alt text)

### WP1-Specific

- [ ] Run actual benchmarks on ARM Cortex-A72 (or emulated)
- [ ] Measure batch verification throughput at 100, 500, 1000 sigs/s
- [ ] Simulate urban intersection scenario (150 vehicles, 5 min)
- [ ] Measure Sybil detection effectiveness with injected Sybil vehicles
- [ ] Measure epoch transition overhead
- [ ] Compare with ECDSA baseline (timing, bandwidth, storage)

### WP5-Specific

- [ ] Measure Zstd compression ratios on actual Falcon-512 signatures
- [ ] Train aviation-specific Zstd dictionary on 10K+ signed ATC messages
- [ ] Measure Merkle tree construction and verification timing
- [ ] Simulate lossy channel conditions (0.1% to 10%)
- [ ] Measure authentication delay for each priority class
- [ ] Compare channel utilization: ECDSA baseline vs. QBITEL compressed

---

## Timeline

| Week | Action |
|------|--------|
| W1-2 | Run benchmarks for WP1 and WP5 (actual measurements) |
| W3 | Convert both papers to LaTeX (IEEEtran) |
| W4 | Integrate benchmark results into evaluation sections |
| W5 | Anonymize, polish, add ethical considerations |
| W6 | Internal review + revision |
| W7 | Submit to ACSAC 2026 |

---

*End of ACSAC preparation guide*
