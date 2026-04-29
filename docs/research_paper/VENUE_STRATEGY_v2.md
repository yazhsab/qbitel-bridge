# QBITEL Whitepaper Series — Revised Venue Strategy v2

**Updated: April 2026**
**Context: CIC reality check applied — crypto-theory venues require formal proofs**

---

## Key Principle: Match Paper Type to Venue Type

| Paper Type | What Reviewers Expect | Your Papers |
|-----------|----------------------|-------------|
| **Crypto-theory** | Formal definitions, reduction proofs, tight bounds | None currently (WP1 after upgrade) |
| **Protocol/systems security** | Threat model, design rationale, implementation, evaluation | WP1, WP2, WP5 |
| **Applied/domain security** | Domain problem, solution architecture, compliance, benchmarks | WP2, WP3, WP5, WP6, WP8 |
| **Privacy engineering** | Privacy definitions, DP accounting, utility analysis | WP4, WP7 |
| **Embedded/constrained crypto** | Resource analysis, optimization, hardware benchmarks | WP3 |

---

## Revised Paper-by-Paper Venue Recommendations

### WP1 — V2X Group Signatures (STRONGEST PAPER)

**Dual-version strategy:**

| Version | Target Venue | What to Add | Timeline |
|---------|-------------|-------------|----------|
| **v1 (Current + minor polish)** | **IET Information Security** (free, Scopus) OR **IEEE TVT** (free non-OA, IF 6.8) | Current version is sufficient for systems-security venues | Submit July 2026 |
| **v2 (CIC-ready)** | **IACR CiC** (free, Diamond OA) | Formal security model (Section 6A), reduction sketches to MLWE/MSIS, adversary game definitions | Submit Oct 2026 |
| **Preprint** | **IACR ePrint + arXiv cs.CR** | As-is | Immediately |

**For v2 CIC upgrade, must add:**
- Formal syntax (KeyGen, Sign, Verify, Open, Judge as algorithms)
- Security games: Full-Anonymity, Full-Traceability, Non-Frameability
- Reduction sketch: Anonymity → MLWE distinguishing advantage
- Reduction sketch: Traceability → MSIS solution extraction
- Epoch-linkability game definition (new contribution)
- VLR soundness game definition

### WP2 — TESLA++ for IEC 61850 (MOST MATURE)

| Priority | Venue | Cost | Indexing | Fit Rationale |
|----------|-------|------|---------|---------------|
| **1st choice** | **Cybersecurity (SpringerOpen)** | **Free** | Scopus, ESCI, DOAJ | Applied security journal; ICS/SCADA is in scope; fast review; Diamond OA |
| 2nd choice | **IEEE Trans. Industry Applications** | Free (non-OA) | Scopus, WoS, IF ~4.2 | IEC 61850 community reads this; strong domain fit |
| 3rd choice | **ACSAC 2026** | Free OA | Scopus, DBLP, CORE A | Applied security conference; good for protocol + systems papers |
| Preprint | **arXiv cs.CR** | Free | Google Scholar | Immediate visibility |

**NOT suitable for:** CIC, TCHES (not enough crypto novelty), USENIX/NDSS (too domain-specific for top-4 security conferences)

### WP3 — Lightweight PQC for Medical Devices

| Priority | Venue | Cost | Indexing | Fit Rationale |
|----------|-------|------|---------|---------------|
| **1st choice** | **IACR TCHES** | **Free** | Scopus, DBLP, CORE A | Embedded crypto implementations; memory/energy benchmarks are exactly what TCHES wants |
| 2nd choice | **ACM TOPS (Trans. Privacy & Security)** | Institutional OA or low APC | Scopus, WoS, DBLP | If framed as privacy for medical IoT |
| 3rd choice | **Cybersecurity (SpringerOpen)** | **Free** | Scopus, ESCI | Broader scope; good fallback |
| 4th choice | **Sensors (MDPI)** | APC ~1800 CHF (waiver possible) | Scopus, WoS, IF ~3.4 | IoT/embedded focus; apply for MDPI waiver |
| Preprint | **arXiv cs.CR** | Free | Google Scholar | |

**Key:** TCHES reviewers want **concrete hardware benchmarks on real constrained platforms**. If you can add MSP430/nRF52840 measurements (not just estimates), TCHES acceptance probability increases significantly.

### WP4 — Homomorphic Vitals Analytics (NEEDS MOST WORK)

| Priority | Venue | Cost | Indexing | Fit Rationale |
|----------|-------|------|---------|---------------|
| **1st choice** | **PoPETs (PETS)** | **Free** | Scopus, DBLP, CORE A | Privacy venue; values system design + DP analysis over crypto novelty |
| 2nd choice | **Healthcare Informatics Research** | **Free** | Scopus, PubMed, DOAJ | Healthcare informatics audience; lower bar for crypto formalism |
| 3rd choice | **Informatics and Health (KeAi)** | **Free until Dec 2026** | Being indexed | Full APC waiver period; good for early publication |
| 4th choice | **JMIR Medical Informatics** | APC ~$2,000 (waiver possible) | Scopus, PubMed, IF ~3.2 | Strong clinical audience |
| Preprint | **arXiv cs.CR** | Free | | |

**Critical for PoPETs:** Must add rigorous DP budget accounting (epsilon composition across queries) and utility-loss experiments. PoPETs reviewers will reject without formal privacy analysis.

### WP5 — ATC Authentication Compression

| Priority | Venue | Cost | Indexing | Fit Rationale |
|----------|-------|------|---------|---------------|
| **1st choice** | **ACSAC 2026** | **Free OA** | Scopus, DBLP, CORE A | Applied security; aviation/bandwidth constraint is a compelling systems story |
| 2nd choice | **Cybersecurity (SpringerOpen)** | **Free** | Scopus, ESCI | ICS/critical infrastructure scope covers aviation |
| 3rd choice | **IEEE Trans. Dependable & Secure Computing** | Free (non-OA) | Scopus, WoS, IF ~7.0 | High-impact; dependability + security angle |
| 4th choice | **IET Information Security** | Free (non-OA) or low APC | Scopus, WoS | Good fit for protocol security papers |
| Preprint | **arXiv cs.CR** | Free | | |

**NOT suitable for:** CIC, TCHES (not enough crypto depth), PoPETs (not a privacy paper)

### WP6 — Multi-Authority Threshold for Banking

| Priority | Venue | Cost | Indexing | Fit Rationale |
|----------|-------|------|---------|---------------|
| **1st choice** | **Financial Cryptography (FC 2027)** | Registration only | Scopus, DBLP, CORE A | Premier venue for crypto + finance intersection |
| 2nd choice | **Cybersecurity (SpringerOpen)** | **Free** | Scopus, ESCI | Covers fintech security |
| 3rd choice | **ACSAC 2026** | **Free OA** | Scopus, DBLP | Applied security angle |
| 4th choice | **Ledger Journal** | **Free** | DOAJ | Niche but free; crypto-finance focus |
| Preprint | **IACR ePrint** | Free | | Crypto audience |

### WP7 — EHR Proxy Re-Encryption

| Priority | Venue | Cost | Indexing | Fit Rationale |
|----------|-------|------|---------|---------------|
| **1st choice** | **PoPETs (PETS)** | **Free** | Scopus, DBLP, CORE A | Privacy-preserving data sharing is core PoPETs scope |
| 2nd choice | **Healthcare Informatics Research** | **Free** | Scopus, PubMed | Healthcare informatics audience |
| 3rd choice | **J. Biomedical Informatics (Elsevier)** | Free (non-OA) | Scopus, WoS, IF ~4.0 | If framed as health IT systems paper |
| 4th choice | **Cybersecurity (SpringerOpen)** | **Free** | Scopus, ESCI | Security angle |
| Preprint | **arXiv cs.CR** | Free | | |

### WP8 — VDF for Industrial Safety

| Priority | Venue | Cost | Indexing | Fit Rationale |
|----------|-------|------|---------|---------------|
| **1st choice** | **ACSAC 2026** | **Free OA** | Scopus, DBLP, CORE A | Novel application of crypto to safety-critical ICS |
| 2nd choice | **Cybersecurity (SpringerOpen)** | **Free** | Scopus, ESCI | ICS security scope |
| 3rd choice | **IEEE Trans. Industry Applications** | Free (non-OA) | Scopus, WoS | IEC 61508 community |
| 4th choice | **IET Smart Grid** | Low APC or promotional free | Scopus, ESCI | Power grid safety angle |
| Preprint | **arXiv cs.CR** | Free | | |

### WP9 — Implicit Certificates for V2X

| Priority | Venue | Cost | Indexing | Fit Rationale |
|----------|-------|------|---------|---------------|
| **1st choice** | **IEEE TVT** | Free (non-OA) | Scopus, WoS, IF ~6.8 | V2X certificate management is core TVT scope |
| 2nd choice | **Vehicular Communications (Elsevier)** | Free (non-OA) | Scopus, WoS, IF ~6.7 | Directly focused on V2X |
| 3rd choice | **NDSS VehicleSec Workshop** | Free | DBLP | Workshop format; lower bar than main NDSS |
| 4th choice | **Cybersecurity (SpringerOpen)** | **Free** | Scopus, ESCI | Fallback |
| Preprint | **IACR ePrint** | Free | | BKE has crypto audience |

---

## Summary: Recommended Submission Plan

### Phase 1 — Immediate (April 2026)

| Action | Papers | Venue | Cost |
|--------|--------|-------|------|
| Post preprints | ALL 9 | IACR ePrint (crypto papers) + arXiv cs.CR (all) | **Free** |

### Phase 2 — First Submissions (May-July 2026)

| Paper | Venue | Cost | Expected Decision |
|-------|-------|------|-------------------|
| WP2 (TESLA++) | Cybersecurity (SpringerOpen) | **Free** | 8-12 weeks |
| WP1 v1 (V2X) | IEEE TVT or ACSAC | **Free** | 12-16 weeks (TVT) or Aug deadline (ACSAC) |
| WP3 (Medical) | IACR TCHES (Issue 1, 2027 deadline ~Sep 2026) | **Free** | 8-12 weeks |
| WP5 (ATC) | ACSAC 2026 | **Free** | Conference notification |
| WP8 (VDF) | ACSAC 2026 | **Free** | Conference notification |

### Phase 3 — Privacy Papers (Aug-Oct 2026, after revision)

| Paper | Venue | Cost | Expected Decision |
|-------|-------|------|-------------------|
| WP4 (HE Vitals) | PoPETs Issue 2, 2027 (deadline ~Jun 2026) | **Free** | 8-12 weeks |
| WP7 (EHR PRE) | PoPETs Issue 3, 2027 (deadline ~Oct 2026) | **Free** | 8-12 weeks |

### Phase 4 — Upgraded Submissions (Oct-Dec 2026)

| Paper | Venue | Cost | What Changed |
|-------|-------|------|-------------|
| WP1 v2 (V2X + formal model) | IACR CiC | **Free** | Added security games, reduction sketches |
| WP6 (Banking) | FC 2027 | Registration | Financial crypto venue |
| WP9 (Implicit Certs) | IEEE TVT | **Free** | |

---

## Total Cost Summary

| Category | Papers | Projected Cost |
|----------|--------|---------------|
| Preprints (ePrint + arXiv) | 9 papers | **$0** |
| Diamond OA journals (Cybersecurity, CiC, TCHES, Healthcare IR) | 4-5 papers | **$0** |
| Free OA conferences (ACSAC, PoPETs) | 3-4 papers | **$0** |
| Subscription-model journals (IEEE TVT, Vehicular Comm) | 2-3 papers | **$0** |
| FC conference registration | 1 paper | ~$500-800 |
| **Total** | **9 papers** | **$0 - $800** |

---

## The Cybersecurity (SpringerOpen) Advantage

This journal deserves special mention as the **safest free option** for most QBITEL papers:

- **Zero APC** (permanently sponsored by Institute of Information Engineering, CAS)
- **Scopus + ESCI (Web of Science)** indexed
- **Impact Factor: 3.7-5.7** (competitive)
- **Rolling submissions** (no deadline pressure)
- **Broad scope:** crypto, network security, ICS, IoT, applied security
- **Fast review:** typically 8-12 weeks
- **SpringerOpen platform:** professional presentation, DOI, full OA

If any first-choice venue rejects, Cybersecurity is a reliable, zero-cost, Scopus-indexed fallback for every single paper in the portfolio.

---

*End of revised venue strategy*
