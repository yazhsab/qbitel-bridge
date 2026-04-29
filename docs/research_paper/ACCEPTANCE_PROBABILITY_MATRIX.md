# QBITEL Papers — Acceptance Probability Matrix

**April 2026 | Based on actual venue statistics + paper quality assessment**

---

## Venue Acceptance Rates (Actual Data)

| Venue | Type | Base Rate | Year | Source |
|-------|------|-----------|------|--------|
| USENIX Security | Conference | 17.1% | 2025 | Official |
| NDSS | Conference | 16.1% | 2025 | Official |
| ACSAC | Conference | 20.7% | 2025 | Official |
| PoPETs | Journal-conf | 26% overall | 2025 | Official |
| FC | Conference | ~21% | 2024 | Official |
| TCHES | Journal | ~25-30% | est. | Community |
| CiC | Journal | Threshold-based | N/A | Quality bar, not quota |
| Cybersecurity (SpringerOpen) | Journal | ~30-40% | est. | Third-party |
| IEEE TVT | Journal | ~30-50% | est. | Third-party |

---

## Per-Paper Acceptance Probability Estimates

These estimates factor in: (1) venue base acceptance rate, (2) paper-venue fit, (3) current paper maturity, (4) competitive landscape for this topic area.

### WP1 — V2X Group Signatures

| Venue | Base Rate | Fit Score | Paper Readiness | **Estimated P(accept)** | Notes |
|-------|-----------|-----------|----------------|------------------------|-------|
| **ACSAC 2026** | 21% | 9/10 | 8/10 | **35-45%** | Strong fit; applied PQC + V2X is timely; epoch-linkability is memorable |
| IEEE TVT | ~35% | 7/10 | 8/10 | **40-50%** | High volume journal; V2X is core scope; less security depth required |
| CiC (v2 with proofs) | Threshold | 7/10 | 5/10 → 8/10 after upgrade | **25-35%** | Needs full proofs; EBL definition is novel enough; risky but high reward |
| NDSS 2027 | 16% | 8/10 | 7/10 | **15-20%** | Very competitive; possible if evaluation is stronger |
| Cybersecurity (SpringerOpen) | ~35% | 8/10 | 8/10 | **50-60%** | Safe fallback; good fit |

**Recommendation chain:** ACSAC (35-45%) → if rejected → IEEE TVT (40-50%) → if rejected → Cybersecurity (50-60%)

### WP2 — TESLA++ for IEC 61850

| Venue | Base Rate | Fit Score | Paper Readiness | **Estimated P(accept)** | Notes |
|-------|-----------|-----------|----------------|------------------------|-------|
| **Cybersecurity (SpringerOpen)** | ~35% | 10/10 | 9/10 | **55-65%** | Excellent fit; ICS + PQC + concrete timing results; fast review (~9 weeks) |
| IEEE TIA | ~30% | 9/10 | 8/10 | **45-55%** | IEC 61850 community reads this; strong domain match |
| ACSAC 2026 | 21% | 8/10 | 8/10 | **30-40%** | Good applied security paper; ICS track exists |
| Computer Networks (Elsevier) | ~25% | 7/10 | 8/10 | **35-45%** | Network protocol angle; free non-OA |

**Recommendation:** Cybersecurity first (55-65%). This is the **highest probability acceptance** across the entire portfolio. Submit immediately.

### WP3 — Lightweight PQC for Medical Devices

| Venue | Base Rate | Fit Score | Paper Readiness | **Estimated P(accept)** | Notes |
|-------|-----------|-----------|----------------|------------------------|-------|
| **IACR TCHES** | ~27% | 9/10 | 6/10 | **20-30%** | Needs real hardware benchmarks (MSP430/nRF52840 cycle counts); high reward |
| ACSAC 2026 | 21% | 7/10 | 7/10 | **25-35%** | Applied security; medical IoT is timely |
| Cybersecurity (SpringerOpen) | ~35% | 8/10 | 7/10 | **40-50%** | Safe option; good fit |
| Sensors (MDPI) | ~40% | 8/10 | 7/10 | **45-55%** | IoT/embedded focus; needs MDPI waiver |

**Recommendation:** Don't rush TCHES. Spend time getting real hardware measurements → then submit TCHES (20-30%). If you can't get hardware: ACSAC (25-35%) or Cybersecurity (40-50%).

### WP4 — Homomorphic Vitals Analytics

| Venue | Base Rate | Fit Score | Paper Readiness | **Estimated P(accept)** | Notes |
|-------|-----------|-----------|----------------|------------------------|-------|
| **PoPETs 2027** | 26% | 8/10 | 4/10 → 7/10 after DP fix | **20-30%** (after revision) | Must add DP budget, utility experiments; strong venue but demanding reviewers |
| Healthcare Informatics Research | ~40% | 9/10 | 6/10 | **40-50%** | Lower bar; healthcare audience; free |
| Informatics and Health (KeAi) | ~50% | 8/10 | 6/10 | **50-60%** | New journal; APC waived until Dec 2026; easier acceptance |
| Cybersecurity (SpringerOpen) | ~35% | 6/10 | 5/10 | **25-35%** | Not ideal fit; needs security framing |

**Recommendation:** Do NOT submit yet. Fix DP rigor first (2-3 weeks work). Then: PoPETs (20-30%) for prestige, Healthcare Informatics Research (40-50%) for safety, or Informatics & Health (50-60%) for quick publication.

### WP5 — ATC Authentication Compression

| Venue | Base Rate | Fit Score | Paper Readiness | **Estimated P(accept)** | Notes |
|-------|-----------|-----------|----------------|------------------------|-------|
| **ACSAC 2026** | 21% | 8/10 | 7/10 | **25-35%** | Aviation security is niche and interesting to reviewers; needs stronger eval |
| Cybersecurity (SpringerOpen) | ~35% | 8/10 | 7/10 | **45-55%** | Good fit; critical infrastructure scope |
| IEEE TDSC | ~20% | 7/10 | 7/10 | **20-25%** | High impact if accepted; free non-OA; competitive |
| IET Information Security | ~35% | 8/10 | 7/10 | **40-50%** | Protocol security focus; reasonable acceptance |

**Recommendation:** ACSAC (25-35%) → if rejected → Cybersecurity (45-55%)

### WP6 — Multi-Authority Threshold Banking

| Venue | Base Rate | Fit Score | Paper Readiness | **Estimated P(accept)** | Notes |
|-------|-----------|-----------|----------------|------------------------|-------|
| **FC 2027** | ~21% | 9/10 | 7/10 | **25-35%** | Premier crypto+finance venue; tiered authorization is novel application |
| Cybersecurity (SpringerOpen) | ~35% | 7/10 | 7/10 | **40-50%** | Fintech security angle |
| Ledger Journal | ~45% | 7/10 | 7/10 | **45-55%** | Niche but free; crypto-finance |
| ACSAC 2026 | 21% | 6/10 | 7/10 | **20-25%** | Less natural fit than FC |

**Recommendation:** FC 2027 (25-35%) for prestige → Cybersecurity (40-50%) as fallback

### WP7 — EHR Proxy Re-Encryption

| Venue | Base Rate | Fit Score | Paper Readiness | **Estimated P(accept)** | Notes |
|-------|-----------|-----------|----------------|------------------------|-------|
| **PoPETs 2027** | 26% | 9/10 | 7/10 | **30-40%** | Privacy-preserving data sharing is core PoPETs; consent model is strong |
| Healthcare Informatics Research | ~40% | 9/10 | 7/10 | **45-55%** | Excellent health IT fit; free |
| J. Biomedical Informatics | ~25% | 8/10 | 7/10 | **30-35%** | Respected health informatics venue; free non-OA |
| Cybersecurity (SpringerOpen) | ~35% | 7/10 | 7/10 | **40-45%** | Security angle; fallback |

**Recommendation:** PoPETs (30-40%) — this paper has the best PoPETs fit of WP4 and WP7. Submit WP7 to PoPETs BEFORE WP4.

### WP8 — VDF for Industrial Safety

| Venue | Base Rate | Fit Score | Paper Readiness | **Estimated P(accept)** | Notes |
|-------|-----------|-----------|----------------|------------------------|-------|
| **ACSAC 2026** | 21% | 8/10 | 7/10 | **25-35%** | Novel application; experimental flag is honest; reviewers will appreciate |
| Cybersecurity (SpringerOpen) | ~35% | 8/10 | 7/10 | **45-55%** | ICS security scope |
| IEEE TIA | ~30% | 9/10 | 7/10 | **40-50%** | IEC 61508 audience; safety engineering angle |

**Recommendation:** ACSAC (25-35%) → IEEE TIA (40-50%) → Cybersecurity (45-55%)

### WP9 — Implicit Certificates V2X

| Venue | Base Rate | Fit Score | Paper Readiness | **Estimated P(accept)** | Notes |
|-------|-----------|-----------|----------------|------------------------|-------|
| **IEEE TVT** | ~35% | 9/10 | 7/10 | **40-50%** | V2X certificate management is core scope; BKE for PQ is timely |
| Vehicular Communications | ~30% | 10/10 | 7/10 | **40-50%** | Directly focused on V2X |
| VehicleSec (NDSS Workshop) | ~30% | 9/10 | 7/10 | **35-45%** | Workshop format; lower bar; good visibility |
| Cybersecurity (SpringerOpen) | ~35% | 6/10 | 7/10 | **35-40%** | Less natural fit |

**Recommendation:** IEEE TVT (40-50%) or Vehicular Communications (40-50%). Both are free non-OA and high-impact for V2X.

---

## Summary: Probability-Ordered Submission Priority

### Highest Probability Acceptances (Submit First)

| Priority | Paper | Venue | **P(accept)** | Cost | Action |
|----------|-------|-------|---------------|------|--------|
| 1 | **WP2** | Cybersecurity (SpringerOpen) | **55-65%** | Free | **Submit immediately** |
| 2 | **WP9** | IEEE TVT | **40-50%** | Free | Submit May 2026 |
| 3 | **WP7** | PoPETs Issue 3, 2027 | **30-40%** | Free | Submit Oct 2026 |
| 4 | **WP1 v1** | ACSAC 2026 | **35-45%** | Free | Submit by ACSAC deadline |

### Medium Probability (Submit Phase 2)

| Priority | Paper | Venue | **P(accept)** | Cost | Action |
|----------|-------|-------|---------------|------|--------|
| 5 | **WP5** | ACSAC 2026 | **25-35%** | Free | Submit with WP1 |
| 6 | **WP8** | ACSAC 2026 | **25-35%** | Free | Submit with WP1 |
| 7 | **WP6** | FC 2027 | **25-35%** | Reg fee | Submit when CFP opens |
| 8 | **WP3** | TCHES | **20-30%** | Free | Only after hardware benchmarks |

### Needs Work Before Submission

| Priority | Paper | Venue | **P(accept)** | Blocker |
|----------|-------|-------|---------------|---------|
| 9 | **WP4** | PoPETs | **20-30%** | DP budget accounting, utility experiments |
| 10 | **WP1 v2** | CiC | **25-35%** | Full reduction proofs (partially done) |

---

## Expected Portfolio Outcome (Realistic)

If you follow this plan and accept fallback to Cybersecurity/IEEE TVT when rejected:

| Scenario | Papers Published | Scopus Indexed | Timeline |
|----------|-----------------|---------------|----------|
| **Optimistic** (60% accept first try) | 6/9 first round + 3 on fallback = **9/9** | 9/9 | 12-18 months |
| **Realistic** (40% accept first try) | 4/9 first round + 5 on fallback = **9/9** | 9/9 | 15-24 months |
| **Conservative** (25% accept first try) | 2/9 first round + 7 on fallback = **9/9** | 9/9 | 18-30 months |

**Key insight:** With Cybersecurity (SpringerOpen) as the universal fallback (free, Scopus, ~35-40% acceptance, rolling), every paper in the portfolio WILL eventually be published. The question is only which tier.

---

*End of acceptance probability matrix*
