# QBITEL Bridge × TCN — Partnership Brief
## Joint Integration Briefing for the VP of Technology

> **Audience:** VP of Technology, TCN (Cloud Contact Center Platform).
> **Purpose:** Present QBITEL Bridge as an embedded security & compliance layer for TCN Operator, with integration architecture, flow diagrams, joint value proposition, and a concrete POC path.
>
> **Version 1.0 | February 2026 | Confidential**

---

## 1. Meeting Objective

To explore a technology partnership in which **QBITEL Bridge becomes an embedded or marketplace-available security & compliance layer inside TCN Operator** — giving TCN's BPO and enterprise contact-center customers a single answer to toll fraud, PCI/HIPAA scope, post-quantum readiness, and per-client multi-tenant compliance, **without any change to the agent experience or TCN's product surface**.

The thirty-second framing:
> *TCN handles the contact center. QBITEL handles what the regulators, auditors, and CISOs ask about the contact center. Together you sell one platform; separately you both leave money on the table.*

---

## 2. What We Understand About TCN Operator

*(Please correct any of the following — this is from public materials only.)*

| Dimension | Our Understanding |
|---|---|
| **Product** | TCN Operator — unified cloud contact center platform |
| **Core capabilities** | Autodialer, manual dialer, preview dialer, IVR, IVM, voicemail delivery, speech analytics, call recording, agent scripting, manager dashboards, live monitoring |
| **Channels** | Inbound, outbound, blended voice, email, SMS |
| **Native integrations** | Salesforce, Zendesk, ServiceNow, Finvi, Zoho, Freshworks, Leadsquared (close to 2,000 logos total) |
| **Integration framework** | Advanced REST API; Synapse engine for outbound webhooks and integration actions (2026 enhancement) |
| **Telephony** | SIP-based voice infrastructure with IVR/IVM logic |
| **Customer base** | BPOs, collections agencies, healthcare, financial services, customer-experience operations |
| **2026 direction** | AI-powered platform enhancements, omnichannel throughput, centralized automation |

The shape we see: TCN owns the dialer, IVR, recording, agent workspace, and CRM/ITSM integration layer. The natural gap — and the one your BPO customers raise loudest — is the **security & compliance layer that sits across all of those components per-client**. That is precisely what QBITEL Bridge is built for.

---

## 3. QBITEL Bridge in One Page

**QBITEL Bridge** is an AI-powered, network-overlay platform that discovers every protocol in a contact-center environment, wraps it in NIST Level 5 post-quantum cryptography, enforces per-tenant compliance and DLP, and generates audit-ready evidence packs — without replacing any existing infrastructure.

### Five modules

| Module | What it produces |
|---|---|
| **Discover** | AI protocol/asset graph — finds SIP, RTP, CTI, IVR, mainframe, CRM API flows in 2–4 hours |
| **Understand** | LLM-generated risk narratives per tenant, per protocol |
| **Modernize** | Auto-generated REST APIs + 6 SDKs for legacy/proprietary protocols (Translation Studio) |
| **Protect** | PQC overlay, DTMF masking, toll-fraud detection, agent DLP, multi-tenant policy enforcement |
| **Prove** | Per-tenant compliance evidence packs (PCI-DSS, HIPAA, SOC 2, GDPR, SOX + 4 more) in <10 min |

### Key technical specs

| Spec | Value |
|---|---|
| Voice PQC overhead | <2 ms (within ITU-T G.114 budget) |
| DTMF masking latency | <5 ms |
| Toll-fraud detection | <1 second |
| Concurrent agent sessions | 20,000+ per deployment |
| Autonomous threat resolution | 78% (no human in the loop) |
| Deployment | 4–6 hours, network overlay, zero downtime |
| LLM inference | Optional on-prem (Ollama) — no customer data egress |

QBITEL is a **complement, not a competitor** to a contact-center platform. We do not run the dialer, the IVR, the agent desktop, or the recording engine. We sit on the wire and at the API boundary, observe what those systems do, and add the controls and evidence that auditors and CISOs require.

---

## 4. The Partnership Thesis

### 4.1 Customer Overlap

Every TCN customer that operates as a BPO — and every TCN customer with PCI, HIPAA, SOX, or GDPR exposure — has a budget line for **exactly the problems QBITEL solves**:

- Toll fraud (industry loses $10B+/year)
- PCI-DSS audit scope (typical BPO spends $500K–$2M/year on audits, up to 80% scope reducible)
- Multi-tenant compliance (BPO running 3+ enterprise clients = 3+ separate annual audits)
- Post-quantum readiness (regulator and large-enterprise client RFP requirement, rising fast)
- Insider data exfiltration (kernel-level DLP for agent endpoints)

Today, those budgets either go unspent (and the BPO absorbs the loss) or go to point tools that do not integrate with TCN. **Embedded in TCN Operator, those budgets flow through TCN.**

### 4.2 Capability Complementarity

| Layer | TCN owns | QBITEL adds |
|---|---|---|
| Voice signaling (SIP) | Trunking, routing, IVR logic | PQC-TLS overlay, toll-fraud pattern matching |
| Voice media (RTP) | Codec, mixing, recording | PQC-SRTP encryption, DTMF masking |
| Agent workspace | Scripting, dashboards | DLP (clipboard, screen, USB, email, voice reading) |
| CRM/ITSM integration | Salesforce/Zendesk/ServiceNow/Zoho native | PII/PHI/PAN masking at API boundary, per-tenant policy |
| Recording storage | Recording engine + retention | Quantum-safe encryption (ML-KEM-1024) + tamper-evident audit |
| Reporting | Operational + analytics dashboards | Compliance evidence packs (9 frameworks) |
| Multi-tenancy | Account/campaign isolation | Cryptographic + policy + audit-trail isolation per BPO client |

No overlap. Two products that obviously belong on the same invoice.

### 4.3 What This Looks Like to the End Customer

A TCN customer logs into TCN Operator as usual. Agents work in the same workspace. Admins see one new tab in the TCN console — *Security & Compliance* — powered by QBITEL. Behind the scenes, every call, every agent session, every CRM API call is being protected, monitored, and evidenced. Compliance reports are downloadable in <10 minutes per framework, per tenant. **No new product surface for the customer to learn.**

---

## 5. Integration Architecture

QBITEL Bridge integrates with TCN Operator at **five well-defined seams**. None of them require changes to the TCN product itself — all are existing TCN integration surfaces or industry-standard network/protocol points.

### 5.1 High-Level Architecture

```
+--------------------------------------------------------------------------+
|                       TCN OPERATOR (CLOUD)                               |
|                                                                          |
|  +------------+   +------------+   +------------+   +-----------------+  |
|  |  Dialers   |   |   IVR /    |   |   Agent    |   |   Recording &   |  |
|  | (Auto /    |   |   IVM      |   | Workspace  |   |   Analytics     |  |
|  |  Preview / |   |   Engine   |   |  Scripting |   |    Storage      |  |
|  |  Manual)   |   |            |   |            |   |                 |  |
|  +-----+------+   +-----+------+   +-----+------+   +--------+--------+  |
|        |                |                |                   |           |
|        +----------------+----------------+-------------------+           |
|                                 |                                        |
|                       +---------+---------+                              |
|                       |  Synapse Webhook  |    REST API                  |
|                       |       Engine      |    Framework                 |
|                       +---------+---------+                              |
|                                 |                                        |
|       Native Integrations:      |                                        |
|       Salesforce / Zendesk /    |                                        |
|       ServiceNow / Zoho /       |                                        |
|       Finvi / Freshworks ...    |                                        |
+---------------------------------+----------------------------------------+
                                  |
                                  |  (1) SIP/RTP mirror via SBC/SPAN tap
                                  |  (2) Synapse webhooks (call events)
                                  |  (3) REST API (recording, agent state)
                                  |  (4) CRM-side hooks (PII masking)
                                  |  (5) Admin SSO + console embed
                                  v
+--------------------------------------------------------------------------+
|                      QBITEL BRIDGE (CO-RESIDENT / CLOUD / ON-PREM)       |
|                                                                          |
|  +----------------+  +----------------+  +-----------------+             |
|  |  AI Discovery  |  |   PQC Data     |  |  Per-Tenant     |             |
|  |  & Protocol    |  |   Plane        |  |  Policy Engine  |             |
|  |  Graph         |  |   (Rust)       |  |  (OPA)          |             |
|  +-------+--------+  +--------+-------+  +--------+--------+             |
|          |                    |                   |                      |
|          +--------+-----------+-------------------+                      |
|                   |                                                      |
|       +-----------+-----------+   +-----------------+                    |
|       |    Compliance         |   |   Threat /      |                    |
|       |    Evidence Engine    |   |   Fraud         |                    |
|       |    (9 frameworks)     |   |   Detection     |                    |
|       +-----------+-----------+   +--------+--------+                    |
|                   |                        |                             |
|                   v                        v                             |
|       Per-tenant audit pack       Autonomous response (78%)              |
|                                                                          |
+--------------------------------------------------------------------------+
```

### 5.2 The Five Integration Seams

| # | Seam | TCN Surface | QBITEL Role |
|---|---|---|---|
| 1 | **SIP/RTP signal & media** | TCN's SBC egress, or a customer-edge SBC mirror | Passive tap or sidecar SBC; injects PQC overlay, performs DTMF masking, runs toll-fraud detection on outbound trunks |
| 2 | **Synapse webhooks** | Existing outbound webhook framework | Subscribes to call.started / call.ended / agent.login / disposition / payment events for compliance evidence + DLP correlation |
| 3 | **REST API** | Existing TCN public API | Pulls recording metadata for quantum-safe encryption tagging; pulls agent state for posture correlation; pulls campaign metadata for per-tenant scoping |
| 4 | **CRM/ITSM hooks** | TCN's native Salesforce / Zendesk / ServiceNow / Zoho / Finvi / Freshworks integrations | QBITEL hooks at the same API boundary to apply field-level PII/PHI/PAN masking before write, and to monitor reads for DLP |
| 5 | **Admin console + SSO** | TCN admin UI | QBITEL exposes a *Security & Compliance* tab inside TCN Operator via iframe + SSO (SAML/OIDC); customer never leaves the TCN console |

**Critical property:** seam #1 is passive — QBITEL is not inline on the voice path by default. If QBITEL is unavailable, calls continue exactly as they do today. Failure mode is open, not closed, unless the customer explicitly opts into inline PQC for trunks they own.

### 5.3 Deployment Topologies

QBITEL supports three deployment topologies for the partnership. Each TCN customer can choose:

| Topology | Where QBITEL runs | Best for |
|---|---|---|
| **TCN-hosted multi-tenant** | Co-located with TCN cloud; QBITEL SaaS instance per TCN region | Small / mid customers; fastest enable |
| **Customer-hosted (on-prem or their cloud)** | Customer's environment; integrates with TCN cloud via SDKs | Regulated customers (defense, government, large bank BPOs) requiring data sovereignty |
| **Hybrid** | Discovery + UI in TCN cloud; data plane + audit packs on customer side | Customers with mixed sensitivity per tenant |

---

## 6. Flow Diagrams — The Three Calls That Matter

### 6.1 Inbound Call With PQC Overlay & Per-Tenant Policy

```
   PSTN/SIP                                              Customer's
   Carrier                                                 CRM
      |                                                     ^
      |  SIP INVITE                                         |
      v                                                     |
+-----------+        +-----------+         +-----------+    |
|   TCN     | -SIP-> |   TCN     | -RTP--> |   TCN     |    |
|   SBC     |        |   IVR     |         |   Agent   |    |
+-----+-----+        +-----+-----+         +-----+-----+    |
      |                    |                     |          |
      |  (1) SPAN/         |  (2) call.started   |  (4) CRM |
      |      mirror        |      webhook        |    write |
      v                    v                     v          |
+----------------------------------------------------------+|
|              QBITEL BRIDGE  (per-tenant scope = tenantA) ||
|                                                          ||
|  Discover protocols      Apply tenant-A policy           ||
|  Verify PQC handshake    Decide: mask DTMF? record? DLP? ||
|  Tag call for evidence   Sign + log every decision       ||
+--------------------+----------------+--------------------+|
                     |                |                     |
                     |  (3) PQC-SRTP  |  (5) Mask           |
                     |  overlay if    |      PII at         |
                     |  trunk eligible|      CRM API        |
                     v                v                     |
                  PQC tunnel       CRM API call ------------+
                  for media
```

**Beat-by-beat:**

1. Carrier delivers SIP INVITE to TCN SBC; TCN SPANs / mirrors the signaling to QBITEL.
2. As TCN's Synapse fires `call.started`, QBITEL receives the webhook, identifies the tenant (from the campaign or DID), and loads that tenant's policy.
3. If the tenant is opted into PQC trunks, QBITEL wraps the media in PQC-SRTP between the carrier-side and TCN-side. Otherwise it stays passive.
4. The agent works the call. When they write to CRM, the call goes through TCN's native integration *plus* QBITEL's mask hook — PAN / SSN / PHI fields are tokenized before persistence.
5. Every decision (DTMF mask fired, recording encrypted, CRM write masked) is signed and logged into the per-tenant evidence stream.

### 6.2 Outbound Call With Toll-Fraud Detection

```
   TCN Operator (Autodialer)
            |
            |  (1) call.started webhook -->  QBITEL
            v
   +-----------------+
   |   TCN SBC       |
   |   (outbound)    |
   +--------+--------+
            |
            |  SIP INVITE to carrier
            v
   +-----------------+
   |  PSTN Carrier   |
   +--------+--------+
            |
   QBITEL inspects in parallel (mirror):
            |
            v
   +--------------------------------------------------+
   |  Toll-fraud engine — 10 pattern detectors        |
   |   * IRSF / premium-rate destination?             |
   |   * Off-hours volume anomaly?                    |
   |   * Compromised trunk pattern?                   |
   |   * Wangiri / call pumping / bypass fraud?       |
   +-----+--------------------------------------+-----+
         |                                      |
   pattern detected (<1s)                  pattern clean
         |                                      |
         v                                      v
   +-----------+                          call proceeds
   |  Action:  |
   |  1. Push  |  -> TCN REST API: campaign.pause
   |  2. Alert |  -> Synapse webhook: noc.alert
   |  3. Log   |  -> per-tenant evidence stream
   +-----------+
```

**Beat-by-beat:**

1. TCN autodialer fires an outbound call. The `call.started` webhook hits QBITEL.
2. QBITEL evaluates the destination + the tenant's recent outbound pattern against 10 fraud-pattern detectors (IRSF, PBX hacking, Wangiri, call transfer fraud, call pumping, bypass, off-hours anomalies, etc.).
3. If a pattern fires, QBITEL calls TCN's REST API to pause the campaign or trunk (subject to tenant policy), pushes a webhook back to the customer's NOC, and writes forensic evidence into the tenant's audit log. Total time: **<1 second from pattern formation to block**.
4. Every block carries a plain-language LLM-generated narrative — not just an alert code.

### 6.3 Payment Call With DTMF Masking & PCI Evidence

```
                 Customer pays by card over voice
                                |
                                v
   +-----------------------------------------------+
   |  Agent says "I'm going to take your card now" |
   +-----------------------+-----------------------+
                           |
                           |  (1) Agent presses "Take Payment"
                           |      in TCN agent workspace
                           v
   +---------------------------+
   |  TCN agent workspace      |
   |  fires payment.start      |
   |  via Synapse webhook      |
   +-------------+-------------+
                 |
                 v
   +---------------------------+
   |  QBITEL receives webhook  |
   |  Apply policy: tenantA    |
   |  PCI-DSS profile          |
   +-------------+-------------+
                 |
                 +--> instruct TCN recording engine: PAUSE
                 +--> instruct TCN media path: CLAMP DTMF tones
                 +--> instruct agent UI: MASK screen fields
                 |
                 v
   Customer enters card via DTMF
   |---- agent hears  CLAMPed tone (no digits)  ----|
   |---- recording   *paused*  for payment span  ----|
   |---- agent screen shows last-4 only           ----|
                 |
                 v
   +---------------------------+
   |  Payment processor        |  (direct path, never touches
   |  receives clean digits    |   TCN or QBITEL after split)
   +-------------+-------------+
                 |
                 |  (2) payment.end webhook
                 v
   +---------------------------+
   |  QBITEL resumes recording |
   |  Generates PCI evidence:  |
   |  - DTMF mask events       |
   |  - Recording pause window |
   |  - Screen mask attestation|
   |  - Signed, append-only    |
   +---------------------------+
```

**The PCI claim:** with this flow active, the agent, the agent's screen, and the TCN recording engine **never see the cardholder data**. PCI scope reduction up to 80%. PCI evidence pack for the tenant generates in <10 minutes on demand.

### 6.4 Multi-Tenant Compliance — The BPO Case

For TCN's BPO customers running multiple downstream enterprise clients (e.g., a 3,000-seat BPO serving a bank on Salesforce, a healthcare payer on Zoho, and a retailer on Zendesk), QBITEL provides:

```
   TCN Operator (one tenant per BPO)
              |
              |  campaign / account-level tagging
              v
   +-------------------------------------------------------+
   |  QBITEL Per-Tenant Logical Partitions                 |
   |                                                       |
   |  +-------------+  +-------------+  +-------------+    |
   |  |  Tenant A   |  |  Tenant B   |  |  Tenant C   |    |
   |  |  (Bank)     |  |  (Health)   |  |  (Retail)   |    |
   |  |             |  |             |  |             |    |
   |  | ML-KEM keys |  | ML-KEM keys |  | ML-KEM keys |    |
   |  | OPA policy  |  | OPA policy  |  | OPA policy  |    |
   |  | PCI + SOX   |  | HIPAA       |  | PCI + GDPR  |    |
   |  | audit log   |  | audit log   |  | audit log   |    |
   |  +-------------+  +-------------+  +-------------+    |
   |                                                       |
   |  Cross-tenant data access is structurally impossible. |
   |  Each tenant's evidence pack contains only that       |
   |  tenant's data. Generation time: <10 min per pack.    |
   +-------------------------------------------------------+
```

This is the single most differentiating capability for TCN's BPO segment — and it's exactly the question TCN's BPO customers are getting in RFPs from their enterprise clients today.

---

## 7. Joint Value Proposition

### 7.1 For TCN

| Benefit | Mechanism |
|---|---|
| **New revenue line per existing customer** | Embedded QBITEL SKU billable per concurrent seat; revenue share with TCN |
| **Higher ACV at quota-bag level** | TCN AEs sell one platform; QBITEL adds materially to ACV (typical 15–30% ARR uplift on a BPO account) |
| **Defensive moat against generic CCaaS competitors** | "TCN is the only CCaaS with embedded NIST Level 5 PQC and multi-tenant compliance automation" — a category-of-one positioning |
| **RFP win-rate lift** | BPO and regulated-enterprise RFPs increasingly include PQC, multi-tenant compliance, toll-fraud SLAs. TCN with QBITEL answers them; TCN without QBITEL has to caveat |
| **Reduced churn on regulated accounts** | Compliance pack delivery removes a recurring customer pain point that often becomes a contract-renewal risk |

### 7.2 For TCN's Customers

| Customer Pain | What They Get |
|---|---|
| Toll fraud surprises on monthly carrier invoice | Real-time blocking, <1s detection, 30–60 day ROI on the QBITEL SKU |
| Annual PCI-DSS audit costing $500K–$2M+ | Up to 80% scope reduction, evidence pack in <10 min |
| Three enterprise clients, three separate audits, no shared evidence | Per-tenant evidence packs, structural isolation |
| Quantum harvest threat against 7-year recording retention | ML-KEM-1024 recording encryption from day one |
| Insider exfiltration of PII via clipboard / USB / screen / voice | Kernel-level DLP across 6 vectors |
| Remote agents on consumer-grade home networks | VPN-less PQC tunnels with continuous posture checks |

### 7.3 For QBITEL

A go-to-market accelerator into a high-fit installed base. TCN's BPO and regulated-enterprise customers are precisely the segments QBITEL targets directly; partnering shortens our sales cycle and scales our deployment footprint.

---

## 8. Commercial Models — Three Options to Discuss

We are flexible on commercial structure and want to find the one that matches how TCN already monetizes integrations.

### Option A — Embedded OEM
- QBITEL ships as a built-in tier of TCN Operator (e.g., *TCN Operator + Bridge Security*).
- TCN bills the customer; QBITEL receives a per-seat revenue share.
- Best for: maximum adoption velocity, simplest customer experience, strongest joint positioning.

### Option B — Marketplace Integration
- QBITEL is listed in TCN's integration marketplace alongside Salesforce, Zendesk, ServiceNow, etc.
- Customers opt in per account; commercial agreement is between QBITEL and the customer.
- TCN receives a referral or platform fee.
- Best for: incremental rollout, lower commitment from TCN, opt-in customer expansion.

### Option C — Referral / Co-Sell
- TCN sales team refers qualified accounts to QBITEL.
- Joint sales motions, shared collateral, separate contracts.
- Referral fee or revenue share per closed deal.
- Best for: a fast first proof-point before deeper commercial integration.

We typically recommend starting at **Option C or B for the first 2 quarters**, validating customer demand and integration quality, then graduating to **Option A** for the long-term partnership.

---

## 9. Technical Risks & Mitigations — The Questions You'll Ask

These are the questions a VP of Technology should ask. Here are our honest answers.

| Risk / Question | Mitigation / Answer |
|---|---|
| *"What happens to call quality if QBITEL is in the media path?"* | PQC overhead is <2ms — within ITU-T G.114 budget. We measure MOS before/after on every deployment. By default we are passive (mirror), not inline. Inline mode is opt-in per trunk per tenant. |
| *"Does this introduce a new failure mode for our voice path?"* | No, in the default topology. Passive mirror means QBITEL failure = no voice impact, calls continue. Inline PQC mode fails open by default; fail-closed is a tenant-level policy choice. |
| *"How does this affect our latency to the Synapse webhook engine?"* | QBITEL subscribes asynchronously. Webhooks fire at TCN's normal latency; QBITEL processes out-of-band. No back-pressure on TCN. |
| *"What's the customer onboarding burden — does our CSM team have to learn a new product?"* | The Security & Compliance tab is iframe + SSO inside TCN Operator. CSM training is a 1-hour session. Per-customer enablement is policy templating, not deployment. |
| *"What about data residency? Some of our customers are in regulated geographies."* | Three deployment topologies (TCN-hosted, customer-hosted, hybrid). On-prem option uses Ollama — zero cloud egress for LLM inference. |
| *"How do we handle a tenant that wants to leave the program?"* | Opt-out is a flag on the tenant. QBITEL stops processing for that tenant within minutes. Tenant's evidence pack is exported and handed over per the contract. No data residue in QBITEL. |
| *"How do we co-engineer the integration without slowing your roadmap or ours?"* | Synapse + REST + SAML/OIDC SSO are existing TCN integration surfaces. We propose a 4-week joint integration sprint with one engineer from each side; deliverable is a working POC in a TCN sandbox. |
| *"What's the liability picture if QBITEL flags toll fraud incorrectly and we pause a campaign?"* | Tenant-configurable: "block + alert," "alert only," or "alert + recommend manual block." Default in early rollout is alert-only; auto-block is opt-in. |
| *"Are you GA-stable? Who runs in production?"* | QBITEL Bridge is in production at BPO and financial-services accounts (references available under NDA). Platform has been hardened against multi-tenant workloads with audit-grade evidence requirements. |
| *"Quantum is 10 years out. Do customers actually buy this today?"* | Two answers: (1) The non-quantum capabilities — toll fraud, DTMF masking, multi-tenant compliance, DLP — pay back the platform in 30–60 days *regardless* of PQC. (2) The harvest-now-decrypt-later threat affects recordings under SOX/HIPAA 7-year retention right now. Both buyers care today. |

---

## 10. Proof of Concept Proposal

We propose a **4-week joint POC** with one TCN engineer, one QBITEL engineer, and one mutually-selected pilot customer (ideally a BPO with PCI exposure or a regulated enterprise account).

### Week 1 — Integration Sprint
- Synapse webhook subscription set up; QBITEL receives `call.started` / `call.ended` / `agent.login` / `disposition` events from a TCN sandbox account.
- REST API auth handshake validated.
- One CRM hook integrated (e.g., Salesforce) for PII masking demonstration.
- SSO (SAML or OIDC) configured for the embedded console tab.

### Week 2 — Capability Activation
- Toll-fraud detection enabled on one outbound campaign; injected test pattern blocks in <1 second.
- DTMF masking enabled on one payment-handling skill; verified card digits never reach recording or agent screen.
- Agent DLP rolled out to 10 pilot agent endpoints.

### Week 3 — Multi-Tenant Demonstration
- Two logical tenants configured in QBITEL inside the single TCN account (representing two of the BPO's downstream clients).
- Per-tenant policy, key, and evidence isolation verified.
- Compliance evidence pack generated for each tenant in <10 minutes.

### Week 4 — Joint Review
- Joint technical readout with TCN engineering, product, and security.
- Customer feedback session with the pilot BPO.
- Decision: proceed to commercial structure (Option A / B / C), expand pilot, or revise scope.

### Success Criteria
- Integration is non-disruptive to TCN's existing data plane (verified by latency + MOS measurement).
- Pilot customer attests that the security/compliance tab is usable inside TCN Operator with no training.
- At least one toll-fraud block, one DTMF-masked payment, and one tenant-specific compliance pack demonstrated end-to-end.
- Joint commercial path identified.

### What We Each Bring

| TCN | QBITEL |
|---|---|
| Sandbox tenant on TCN Operator | Bridge platform instance (TCN-hosted or QBITEL-hosted, mutual choice) |
| One engineer (4 weeks, part-time) | One integration engineer (4 weeks, full-time) |
| Synapse / REST / SSO documentation access | All POC software, no charge |
| Pilot customer introduction | All deployment and operational support |
| Product / security stakeholder for the joint readout | Solution architect for the joint readout |

---

## 11. Next Steps

If today's conversation is productive, the next steps are simple:

1. **Within 1 week** — Mutual NDA in place; QBITEL shares the technical reference architecture under NDA, plus customer references.
2. **Within 2 weeks** — Joint technical scoping call with TCN engineering + QBITEL integration team to confirm the Synapse / REST / SSO seams and pick a pilot customer.
3. **Within 4 weeks of scoping call** — POC kickoff per Section 10 above.
4. **Within 12 weeks** — Joint go/no-go decision on commercial structure and roadmap to GA partnership.

### What we'd like from this meeting
- Your initial reaction to the integration architecture in Section 5.
- Confirmation (or correction) of our understanding of TCN Operator in Section 2.
- A pointer to the right TCN counterpart for the Week-1 technical scoping call (likely an integration architect + a product manager).
- An indication of which commercial model (Section 8) is most natural for TCN's current partner motion.

---

## Appendix A — Quick Reference Cards

### A.1 The Five Integration Seams

| # | Seam | Direction | Existing TCN surface? |
|---|---|---|---|
| 1 | SIP/RTP signal & media | TCN → QBITEL (mirror); optional QBITEL → TCN (inline PQC) | SBC mirror is standard |
| 2 | Synapse webhooks | TCN → QBITEL | Existing 2026 webhook engine |
| 3 | REST API | QBITEL → TCN (pull); TCN → QBITEL (push) | Existing public API |
| 4 | CRM/ITSM hooks | QBITEL alongside TCN at CRM API boundary | Hooks existing native integrations |
| 5 | Admin console + SSO | TCN console iframe → QBITEL UI | Standard SAML/OIDC |

### A.2 Compliance Frameworks QBITEL Automates Inside TCN

PCI-DSS 4.0 · TCPA · HIPAA · HITECH · SOC 2 Type II · GDPR · SOX · GLBA · FCA/MiFID II · NIST PQC

### A.3 What QBITEL Does *Not* Do

- We do not replace any TCN component
- We do not run the dialer, IVR, agent workspace, or recording engine
- We do not introduce a new product surface for the agent or end customer
- We do not require TCN to change its roadmap or product architecture
- We do not require customers to deploy new endpoint software on-call (DLP is optional and per-tenant)

### A.4 Companion Documents

| Document | When useful |
|---|---|
| QBITEL Bridge — BPO Product Owner Walkthrough | Hand to TCN's product / solution-architecture team |
| QBITEL Bridge — BPO Sales Manager Walkthrough | Hand to TCN's sales leadership |
| QBITEL Bridge — BPO Marketing Pitch | For TCN's marketing team if joint co-marketing is in scope |
| QBITEL Bridge — BPO Q&A Guide | RFP-response source material for TCN's bid team |

---

*QBITEL Bridge × TCN — Partnership Brief.*
*Confidential — For Authorized Recipients Only — © 2026 QBITEL.*
*enterprise@qbitel.com | bridge.qbitel.com*
