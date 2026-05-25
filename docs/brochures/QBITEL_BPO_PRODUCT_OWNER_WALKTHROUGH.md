# QBITEL Bridge — Product Owner Walkthrough
## BPOs & Call Centers Running Multiple Client Stacks

> **Audience:** Product Managers, Solution Architects, IT Product Owners, Engineering Leads at BPOs evaluating whether QBITEL Bridge fits a portfolio of heterogeneous client environments.
>
> **Version 1.0 | February 2026 | Confidential**

---

## How to Read This Document

This is a sequential, evaluation-grade walkthrough of QBITEL Bridge — what it does, how it's built, how it lands on your existing infrastructure, and most importantly, **how it handles the reality that every client in your BPO mandates a different dialer, CRM, and ticketing stack**.

If you read nothing else, read **Section 5: Multi-Client / Multi-Stack Coverage Matrix**. That section directly answers the question every BPO product owner asks: *"Can one platform really sit across Avaya + Genesys + Asterisk simultaneously, with separate Salesforce/Zoho/HubSpot CRMs and separate ServiceNow/Freshdesk/Zendesk ticketing, and keep each client's data, policies, and audits totally isolated?"*

---

## 1. Executive Snapshot

**QBITEL Bridge** is an AI-powered, network-overlay security and modernization platform built for environments full of legacy and heterogeneous infrastructure. It is positioned as a **Critical Systems Intelligence and Protection Platform** — designed to *discover, understand, modernize, protect, and prove* control over systems that cannot easily be replaced.

For a BPO, that translates to one sentence:

> **QBITEL sits at the network layer of your contact center, discovers every dialer/CRM/ticketing protocol your clients run, wraps them in quantum-safe encryption, applies per-tenant policies and DLP, and produces per-client compliance evidence — without replacing a single piece of your existing infrastructure.**

**The five modules at a glance:**

| Module | What it produces for a BPO |
|---|---|
| **Discover** | A live protocol & asset graph spanning every client's dialer, CRM, ticketing, and recording system |
| **Understand** | Per-client risk narratives, fraud-pattern analysis, insider-threat signals |
| **Modernize** | Auto-generated REST APIs and SDKs that wrap legacy protocols (TN3270e mainframe sessions, SS7, custom CTI) |
| **Protect** | Post-quantum encryption overlay, DTMF masking, toll-fraud blocking, agent DLP, per-tenant policy enforcement |
| **Prove** | Per-tenant compliance evidence packs (PCI-DSS, HIPAA, SOC 2, SOX, GDPR, etc.) generated in <10 minutes |

**Deployment:** 4–6 hours, zero downtime, network-overlay tap. No PBX replacement. No agent retraining. No CRM disruption.

---

## 2. The BPO Reality We're Designed For

A typical mid-to-large BPO is not a single tech stack. It is a **portfolio of client-mandated stacks running simultaneously on the same floor (and the same WFH agents)**. Every enterprise client brings their security policy, their preferred dialer, their CRM of record, and their ticketing system. The BPO's job is to operate all of them concurrently while passing PCI-DSS / HIPAA / SOC 2 audits — separately, for each client.

For the rest of this document, we will walk through QBITEL using one composite anchor scenario.

### The Anchor Scenario — Acme Outsourcing (3,000 seats + 600 WFH)

| Client | Industry | Dialer / Telephony | CRM | Ticketing | Primary Compliance |
|---|---|---|---|---|---|
| **NorthBank** | Banking — credit card disputes & collections | Avaya Aura CM + Cisco UCCE | Salesforce Financial Services Cloud | ServiceNow ITSM | PCI-DSS 4.0, SOX, FCA |
| **CareFirst Health** | Healthcare payer — claims & member services | Genesys Cloud CX | Zoho CRM Plus | Freshdesk | HIPAA, HITECH |
| **ShopRight Retail** | E-commerce — order support & returns | Asterisk / FreePBX | HubSpot Service Hub | Zendesk Support | PCI-DSS, GDPR, TCPA |

Plus a 600-agent work-from-home pool split across all three clients, accessing client-owned systems from home networks.

This is the scenario the rest of this walkthrough will keep returning to. Every capability we describe is mapped against *all three* client stacks — because in a BPO, that's the only test that matters.

---

## 3. Architecture in 4 Layers

QBITEL Bridge is a four-layer polyglot platform. Each layer is built in the language best suited to its job.

| Layer | Language | Responsibility |
|---|---|---|
| **UI Console** | React / TypeScript | Admin dashboard, Protocol Copilot, multi-tenant policy editor, marketplace |
| **Control Plane** | Go | Service orchestration, OPA policy engine, Vault secret management, gRPC API gateway |
| **AI Engine** | Python (FastAPI) | Protocol discovery, LLM inference (on-prem Ollama option), anomaly detection, compliance reasoning |
| **Data Plane** | Rust | Wire-speed PQC-TLS encryption, DPDK packet processing, DTMF masking, toll-fraud pattern matching |

### Deployment Model — Network-Overlay, Not Inline Replacement

QBITEL does **not** sit between your agents and the dialer. It sits **alongside** — taking a passive copy of the traffic from a SPAN port / network tap, then injecting PQC-protected tunnels at the protocol layer.

The implication for a BPO product owner:
- **No latency added to the existing voice path** — the data plane operates on a mirror, then steers protected traffic through a PQC overlay.
- **No single point of failure** — if QBITEL fails open, calls continue on the underlying PBX/dialer exactly as they do today.
- **No vendor lock-in** — pulling QBITEL out leaves the original infrastructure untouched.

### Deployment Options

| Option | When to use | Notes |
|---|---|---|
| **Docker Compose** | Pilot, single-site lab | Fastest start, <10 minutes to running |
| **Kubernetes / Helm** | Production multi-site | Ships with Prometheus, Grafana, Jaeger observability |
| **Air-gapped on-prem** | Defense-grade BPOs, regulated client mandates | Ollama-based local LLM, zero cloud egress |
| **Hybrid** | Multi-site with mixed sensitivity | Per-site policies, central observability |

For Acme Outsourcing, we'd typically recommend Kubernetes at the primary site with a per-tenant logical control plane (more on this in Section 5).

---

## 4. Walkthrough — Module by Module, Applied to Acme

### 4.1 Discover — Asset & Protocol Graph

**What it does:** AI-driven discovery scans all mirrored network traffic and builds a live graph of every protocol, every endpoint, every flow, and every cryptographic state present in the environment.

**Discovery pipeline:**

| Stage | Duration | Output |
|---|---|---|
| Statistical analysis | 5–10 seconds | Entropy, byte frequency, binary vs. text classification |
| ML classification (CNN + BiLSTM) | 10–20 seconds | Protocol family identification, 89%+ first-pass accuracy |
| PCFG grammar inference | 1–2 minutes | Field structure, message types, semantic learning |
| Parser generation | 30–60 seconds | Auto-generated parsers at 50,000+ messages/sec |
| Continuous learning | Ongoing | Refines grammar as new patterns appear |

**What this produces for Acme:** In a single 2–4 hour pass, the discovery engine identifies — across all three client environments simultaneously — SIP/SDP on NorthBank's Avaya trunks, the proprietary CTI variant on Cisco UCCE, the Genesys Cloud WebRTC media flows for CareFirst, Asterisk AMI/ARI events for ShopRight, plus the TN3270e mainframe sessions NorthBank agents use, the Salesforce REST/streaming API traffic, Zoho REST flows, HubSpot service flows, and ServiceNow/Freshdesk/Zendesk ticket-system traffic.

It also discovers what your own IT team didn't know was there — undocumented legacy protocols, shadow integrations, end-of-life endpoints still in production. The output is a single asset and protocol graph, **tagged per tenant**, that becomes the foundation for every other module.

### 4.2 Understand — Risk Narratives

The Understand module takes the discovery graph and produces human-readable risk explanations using on-premise LLM reasoning. Every conclusion carries confidence and evidence (per the platform's positioning guardrails — "every AI conclusion must carry confidence and evidence").

**Examples of what Acme's product owner would see:**

- *"NorthBank — Avaya trunk T-04 shows unencrypted SIP carrying DTMF tones in RFC 2833 format during the 14:00–17:00 window when 67% of card-not-present transactions occur. Estimated PCI scope exposure: 1,247 calls/day. Recommended: enable DTMF masking + SIP-PQC overlay for this trunk."*
- *"CareFirst — Genesys Cloud agent endpoints in the WFH pool show 23 sessions/day where PHI fields are visible on screen for >30 seconds during inactivity. Recommended: enforce session-timeout policy + clipboard DLP."*
- *"ShopRight — Asterisk SIP trunk shows 14 calls in the last 6 hours to numbers in 4 high-risk premium-rate country codes. Pattern matches IRSF profile. Estimated weekend exposure if not blocked: $42,000."*

This is the module that lets a product owner have a *real conversation* with the platform — not a dashboard full of alert codes.

### 4.3 Modernize — Translation Studio

For Acme's NorthBank operations, agents access a legacy mainframe via TN3270e. Modernizing that is normally a multi-year, multi-million-dollar project.

QBITEL's **Translation Studio** takes a discovered protocol and auto-generates:

- A REST/gRPC API surface that mirrors the protocol semantics
- Six SDKs (Python, Node.js, Java, .NET, Go, Rust)
- OpenAPI specifications and integration test harnesses
- A replay/test sandbox to validate behavior before production cutover

For a BPO product owner, the practical outcome is: **NorthBank's mainframe screens can be wrapped in a REST API that the agent desktop, Salesforce, or a future bot can call** — without touching the mainframe itself. The Protocol Marketplace (next sub-section) ships pre-built adapters for the most common legacy systems; Translation Studio handles the long tail of proprietary client systems.

### 4.4 Protect — Runtime Controls

This is where most of QBITEL's BPO-specific value lands. The Protect module applies six runtime controls, all enforced at wire speed by the Rust data plane:

| Control | What it does | Latency budget |
|---|---|---|
| **PQC overlay** | NIST Level 5 post-quantum encryption (ML-KEM + ML-DSA) wrapped around SIP/RTP/SS7/TN3270e | <2ms voice overhead, within ITU-T G.114 budget |
| **DTMF masking** | Real-time CLAMP / FLAT_TONE / SILENCE on payment digits — in headset AND recording | <5ms |
| **Toll-fraud engine** | 10 fraud-pattern detectors (IRSF, PBX hacking, Wangiri, call transfer fraud, call pumping, bypass fraud, etc.) | <1 second pattern detection |
| **Agent DLP** | Six exfiltration vectors blocked (clipboard, screen capture, USB, email/chat, voice reading, screen scraping) | Kernel-level, real-time |
| **Per-tenant policy enforcement** | OPA policies applied separately per client tenant | Real-time at the control plane |
| **Remote agent tunnels** | VPN-less PQC tunnels (ML-KEM-768 + AES-256-GCM) with continuous endpoint posture | <200ms tunnel setup |

For Acme, this means NorthBank gets full PCI controls + SOX recording integrity, CareFirst gets HIPAA-grade encryption and PHI DLP, and ShopRight gets PCI for payment flows + GDPR consent enforcement — **with separate keys, separate policies, separate audit trails**, all from the same QBITEL instance.

### 4.5 Prove — Per-Tenant Evidence Packs

The Prove module continuously gathers evidence (call records, control events, policy decisions, cryptographic operations) and generates compliance reports on demand.

**Nine frameworks supported:**

| Framework | Relevant to |
|---|---|
| PCI-DSS 4.0 | NorthBank, ShopRight |
| TCPA | ShopRight outbound dialing |
| HIPAA | CareFirst |
| HITECH | CareFirst |
| SOC 2 Type II | All three (Acme as service org) |
| GDPR | ShopRight EU customers |
| SOX | NorthBank financial recording |
| GLBA | NorthBank financial data |
| FCA/MiFID II | NorthBank if UK/EU |
| NIST PQC | All — quantum-safe transition evidence |

Each tenant gets its own evidence pack. Acme can hand NorthBank a PCI-DSS report that contains *only NorthBank's data*, CareFirst a HIPAA report that contains *only CareFirst's data*, and ShopRight a PCI+GDPR report that contains *only ShopRight's data* — all generated in **<10 minutes per framework, per tenant**, on demand, with blockchain-backed tamper evidence.

This single capability typically retires entire compliance-evidence-collection teams for a BPO.

---

## 5. Multi-Client / Multi-Stack Coverage Matrix
### *(The section every BPO product owner should read twice)*

This is the core of the document. The question is not "does QBITEL support Avaya?" The question is "does QBITEL support **all the dialers, CRMs, and ticketing systems my clients mandate, simultaneously, with isolation between clients?**"

The answer is yes. Here is how.

### 5.1 Dialer / Telephony Coverage

| Client | Platform | QBITEL Integration | What QBITEL Protects |
|---|---|---|---|
| NorthBank | Avaya Aura CM + Cisco UCCE | TSAPI / DMCC for Avaya; CTI-OS / Finesse API for Cisco; both wrapped in PQC tunnel | SIP/SDP signaling, RTP/SRTP media, CTI events, recording streams |
| CareFirst | Genesys Cloud CX | REST API + PQC-TLS; WebRTC media leg encrypted via PQC-SRTP | Signaling, media, agent state events, recording API |
| ShopRight | Asterisk / FreePBX | AMI/ARI with PQC tunnel | SIP trunks, RTP media, dialplan execution, outbound trunk control for toll-fraud blocking |
| (Any new client) | Any PBX / SBC / Cloud CC | Protocol-level overlay; AI discovers, marketplace provides parser | New protocols added without infrastructure changes |

### 5.2 CRM Coverage

| Client | CRM | QBITEL Integration | Controls Applied |
|---|---|---|---|
| NorthBank | Salesforce Financial Services Cloud | REST API + Streaming API, PQC-TLS, PII masking middleware | Card data masked at API boundary; PAN detection on inbound writes; screen-rendered PII shows last 4 digits only |
| CareFirst | Zoho CRM Plus | REST API + PQC-TLS | PHI fields tagged and DLP-monitored; outbound exports blocked when PHI patterns detected |
| ShopRight | HubSpot Service Hub | REST API + PQC-TLS | PCI-tagged fields masked; GDPR consent state tracked and propagated to recordings |

QBITEL does not replace any CRM. It sits at the network/API integration boundary, applies data classification and masking as data flows between dialer ↔ CRM, and produces evidence that the right field-level controls fired on the right calls.

### 5.3 Ticketing / ITSM Coverage

| Client | Ticketing System | QBITEL Integration | Controls Applied |
|---|---|---|---|
| NorthBank | ServiceNow ITSM | REST API + PQC-TLS, workflow webhooks | Auto-create incidents on fraud/policy events; tamper-evident incident audit trail; PII scrubbing on ticket bodies |
| CareFirst | Freshdesk | REST API + PQC-TLS | PHI detection on ticket creation; HIPAA-compliant retention policy enforced; access logged for audit |
| ShopRight | Zendesk Support | REST API + PQC-TLS | PCI-tagged conversation transcripts; GDPR DSAR automation; right-to-erasure workflow |

QBITEL's evidence pipeline reads these systems' audit logs and folds them into the compliance evidence pack — so when NorthBank's auditor asks "show me every ticket that handled cardholder data and prove it was access-controlled," that answer is one query.

### 5.4 Per-Tenant Isolation — How It Actually Works

This is the question with the highest stakes for a BPO. The answer is layered:

| Isolation Layer | Mechanism |
|---|---|
| **Cryptographic keys** | Separate ML-KEM key material per tenant; compromise of one tenant's keys has zero blast radius to others. Keys backed by Vault, optionally HSM. |
| **Policy** | OPA policies scoped to a tenant ID; the policy engine refuses to evaluate cross-tenant. |
| **Network** | Per-tenant logical network segments enforced at the protocol layer — a NorthBank agent's session cannot egress to a CareFirst destination. |
| **Audit trail** | Append-only, per-tenant audit logs with blockchain-backed integrity. Auditor for tenant A cannot see tenant B's evidence. |
| **Identity** | Per-tenant agent identity scoping; an agent assigned to NorthBank cannot authenticate to CareFirst-tagged systems without explicit re-provisioning. |
| **Compliance reports** | Generated against a single tenant scope; the report generator literally cannot pull data outside the requested tenant. |

In Acme's three-client scenario, this means the PCI-DSS evidence pack handed to NorthBank's QSA does not contain — and cannot contain — any reference to CareFirst's PHI or ShopRight's GDPR data.

### 5.5 "What happens when our 4th client arrives with a stack we haven't seen?"

This is the long-tail problem for every BPO. QBITEL handles it with two mechanisms:

**1. Protocol Marketplace (1,000+ pre-built protocols).** Before any new-client onboarding, check the marketplace. Most enterprise dialers, CRMs, ITSM tools, mainframe protocols, and payment systems are already there.

**2. AI Discovery + Translation Studio (for the unknown).** If a new client brings a proprietary or custom protocol, the discovery engine finds it within 2–4 hours, the AI engine infers the grammar, and Translation Studio generates a parser and adapter. Time-to-coverage for a brand new protocol family is typically days, not months.

The practical implication: **adding a 4th, 5th, or Nth client to Acme's QBITEL deployment is an incremental policy/onboarding exercise, not a re-architecture.**

---

## 6. Core Capabilities — Deep Dive (for Evaluation)

This section is intended for technical evaluators who want to test specific claims. For each capability we list what it does, the testable metric, and how a product owner would validate it in a POC.

### 6.1 AI Protocol Discovery

- **Claim:** Discovers all voice/data/CTI protocols across multiple client stacks in 2–4 hours, with 89%+ first-pass accuracy on protocol family classification.
- **How to test:** Place a passive tap on a single Acme floor running all three clients. Ask QBITEL to produce the discovered protocol graph within 4 hours. Cross-check against a hand-built inventory from your IT team. Look for the *unknowns* it surfaces.
- **Key metric:** Time to complete graph; count of previously-undocumented protocols found.

### 6.2 Translation Studio

- **Claim:** Auto-generates REST/gRPC APIs and 6 SDKs for any discovered protocol, including legacy mainframe (TN3270e).
- **How to test:** Pick one of NorthBank's TN3270e screens. Have QBITEL generate a REST adapter. Call it from a Salesforce Apex test. Verify round-trip and replay sandbox behavior.
- **Key metric:** Time from "select protocol" to "first successful API call"; SDK quality (idiomatic, typed, documented).

### 6.3 Post-Quantum Cryptography (PQC)

- **Claim:** NIST Level 5 PQC (ML-KEM-1024 + ML-DSA-87 for payment/recording; ML-KEM-512 + Falcon-512 for voice signaling/media; ML-KEM-768 + ML-DSA-65 for tunnels). Voice overhead <2ms.
- **How to test:** Compare MOS scores and end-to-end latency on a representative call sample with QBITEL on vs. off. Verify against ITU-T G.114 budget.
- **Key metric:** Added latency on signaling and media; PQC algorithm exposed in handshake logs.

### 6.4 Agentic AI Security — 78% Autonomous Resolution

- **Claim:** 78% of routine security events resolved without human intervention; each automated action carries a plain-language LLM-generated narrative; LLM inference can run fully on-premise via Ollama.
- **How to test:** Run a structured incident simulation (toll-fraud pattern injection, simulated insider exfiltration, simulated SIP injection). Measure auto-resolution rate and response time. Inspect the narrative output for each.
- **Key metric:** Time to resolution per event category; ratio of auto-resolved to escalated; quality of narrative.

### 6.5 Multi-Tenant Compliance Automation

- **Claim:** Generate per-tenant evidence packs across 9 frameworks in <10 minutes per framework. Cross-tenant data exfiltration is structurally impossible.
- **How to test:** Generate a PCI report for NorthBank and a HIPAA report for CareFirst simultaneously. Inspect both for any cross-tenant data. Time each generation. Verify blockchain-backed audit trail integrity.
- **Key metric:** Generation time per framework; zero cross-tenant references; audit-trail tamper evidence.

### 6.6 Remote Agent Tunnels — VPN-less

- **Claim:** ML-KEM-768 + AES-256-GCM PQC tunnel direct from agent endpoint to QBITEL gateway, with continuous posture (OS version, AV, disk encryption, WiFi WPA3, geo-fence).
- **How to test:** Deploy the agent client to a sample WFH machine. Validate posture checks. Try to bypass with split tunneling — verify QBITEL blocks. Test from out-of-policy geography.
- **Key metric:** Tunnel setup time; posture-check enforcement; deny rate on simulated bypass attempts.

---

## 7. Deployment & Operational Model

### 7.1 The 4-Step Deployment

| Step | Duration | Activity |
|---|---|---|
| 1. Network tap | 30 minutes | Non-invasive SPAN port / passive tap on voice and data networks |
| 2. Protocol discovery | 2–4 hours | AI identifies every protocol, every flow, every endpoint across all client environments |
| 3. Encryption activation | 1 hour | PQC overlays activated for all discovered protocols |
| 4. Policy deployment | 30 minutes | Per-tenant security and compliance policies configured |
| **Total** | **4–6 hours** | **Full quantum-safe protection, zero downtime, nothing replaced** |

### 7.2 Observability — What the BPO's NOC Sees

QBITEL ships with Prometheus metrics, Grafana dashboards, and Jaeger tracing out of the box. For Acme's NOC, the relevant views are:

- **Per-tenant operational dashboard** — call volumes, encrypted vs. unencrypted, fraud blocks, DLP events
- **Per-tenant compliance dashboard** — control coverage, evidence completeness, upcoming audit windows
- **Cross-tenant platform health** — discovery engine load, AI engine inference latency, data-plane throughput
- **Incident timeline** — autonomous actions with their LLM-generated narratives

### 7.3 What QBITEL Automates vs. What the BPO Owns

| Owned by QBITEL platform | Owned by Acme's NOC / IT |
|---|---|
| Protocol discovery, classification, parser generation | SPAN port / tap provisioning |
| Encryption-key lifecycle (rotation, revocation) | Active Directory / IdP integration |
| Toll-fraud pattern matching and trunk blocking | Trunk and SBC carrier configuration |
| Agent DLP enforcement | Endpoint MDM enrollment |
| Compliance evidence collection and report generation | Final auditor handoff and explanation |
| 78% of routine security events | The other 22% — escalated exceptions with full LLM-generated context |
| Per-tenant policy enforcement | Per-tenant policy authoring (with QBITEL templates) |

---

## 8. Integration Effort by Client Stack

A realistic estimate for onboarding each of Acme's clients onto QBITEL once the platform is deployed:

| Client | Stack | Integration effort | Notes |
|---|---|---|---|
| NorthBank | Avaya + Cisco + Salesforce + ServiceNow + TN3270e mainframe | 3–5 business days | All four primary systems are pre-built in the Protocol Marketplace. Mainframe modernization via Translation Studio is optional but recommended for the long term. |
| CareFirst | Genesys Cloud + Zoho + Freshdesk | 2–3 business days | All three are pre-built. HIPAA policy template applied per tenant. WFH agent tunnel rollout is the longer-tail item. |
| ShopRight | Asterisk + HubSpot + Zendesk | 2–3 business days | All three are pre-built. GDPR DSAR automation worth scoping into the engagement. |
| **Total** | All three concurrent | **~2 weeks** | Plus 30 days of post-go-live monitoring (standard QBITEL CSM engagement). |

If a fourth client arrives with a stack mostly in the marketplace, expect 2–4 days. If they arrive with one or more proprietary protocols, add 1–2 weeks for Translation Studio adapter generation and validation.

---

## 9. Roadmap & Extensibility

The product is structured so that the BPO never has to wait for the QBITEL roadmap to support a new client's stack. Three extensibility paths:

- **Protocol Marketplace** — community and QBITEL-published adapters. Acme can submit Acme-built adapters back if they choose to monetize their own work.
- **Translation Studio** — for proprietary client protocols, the BPO can generate adapters in-house without QBITEL professional services.
- **Policy templates** — every compliance framework ships with a default OPA policy. Acme's compliance team can fork and customize per client without changing the platform.

The published roadmap (see internal QBITEL product strategy briefings) is heaviest on (a) more pre-built marketplace protocols, (b) more compliance frameworks beyond the current nine, and (c) deeper Translation Studio coverage for niche legacy systems.

---

## 10. Evaluation Checklist for the Product Owner

If you're heading into a POC, take this list with you. Each item is a concrete demonstrate-X demand.

- [ ] Run protocol discovery against a live mirror of one Acme floor. Produce the discovered graph within 4 hours.
- [ ] Surface at least three previously-undocumented protocols, endpoints, or flows.
- [ ] Stand up three logical tenants (NorthBank / CareFirst / ShopRight) and demonstrate cryptographic key separation.
- [ ] Activate DTMF masking on one trunk; verify card digits absent in both the agent headset capture and the recording.
- [ ] Inject a simulated IRSF pattern; verify auto-block within 1 second and a generated NOC narrative.
- [ ] Trigger a simulated PHI exfiltration via clipboard from a CareFirst-tagged agent; verify block and audit entry.
- [ ] Generate a PCI-DSS evidence pack for NorthBank in <10 minutes; verify zero CareFirst/ShopRight data in the pack.
- [ ] Generate a HIPAA evidence pack for CareFirst in <10 minutes; verify zero NorthBank/ShopRight data in the pack.
- [ ] Deploy one WFH agent tunnel; validate posture checks; attempt a split-tunnel bypass and verify denial.
- [ ] Auto-generate a REST adapter for one TN3270e screen via Translation Studio; call it from a Salesforce sandbox.
- [ ] Walk a sample auditor through one tenant's evidence pack end-to-end; capture their feedback.
- [ ] Run the full deployment in 4–6 hours, end-to-end, with zero impact on live call quality (verify with MOS sampling).

If the platform passes those twelve items, it covers every claim made in this walkthrough.

---

## Appendix A — Where to Read Further

| Topic | Source |
|---|---|
| Vertical brochure (technical features) | [docs/brochures/10_BPO_CALL_CENTERS.md](10_BPO_CALL_CENTERS.md) |
| Deployment & delivery checklist | [docs/brochures/QBITEL_BPO_DEPLOYMENT_CHECKLIST.md](QBITEL_BPO_DEPLOYMENT_CHECKLIST.md) |
| Architecture reference | [architecture.md](../../architecture.md) |
| Product positioning | [docs/QBITEL_FINAL_PRODUCT_POSITIONING.md](../QBITEL_FINAL_PRODUCT_POSITIONING.md) |
| Sales-facing companion | [docs/brochures/QBITEL_BPO_SALES_MANAGER_WALKTHROUGH.md](QBITEL_BPO_SALES_MANAGER_WALKTHROUGH.md) |
| Pitch Q&A guide | [docs/brochures/QBITEL_BPO_PITCH_QA_GUIDE.md](QBITEL_BPO_PITCH_QA_GUIDE.md) |

---

*QBITEL Bridge — Product Owner Walkthrough for BPO & Call Center Environments.*
*Confidential — For Authorized Recipients Only — © 2026 QBITEL.*
