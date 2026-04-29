# QBITEL Bridge — BPO Pitch Q&A Guide
## Questions & Answers for Business and Product Teams

> **Purpose:** Prepare sales, presales, and product teams for tough questions from BPO CXOs, IT/Security leaders, Compliance officers, and Procurement teams during pitches, demos, and RFP responses.

---

## HOW TO USE THIS GUIDE

- **Before a pitch:** Review the audience section matching your buyer persona
- **During a demo:** Use the one-liners for quick responses; expand with detail if pressed
- **RFP responses:** Use the detailed answers as base text
- **Objection handling:** See Section 7 for the hardest pushbacks

---

## TABLE OF CONTENTS

1. [Business & ROI Questions (CXO / CFO)](#1-business--roi-questions)
2. [Technical & Architecture Questions (CTO / IT Director)](#2-technical--architecture-questions)
3. [Security & Risk Questions (CISO / Security Team)](#3-security--risk-questions)
4. [Compliance & Legal Questions (Compliance Officer / Legal)](#4-compliance--legal-questions)
5. [Operations Questions (Contact Center Director / COO)](#5-operations-questions)
6. [Procurement & Vendor Questions (Procurement / Vendor Management)](#6-procurement--vendor-questions)
7. [Hard Objections & How to Handle Them](#7-hard-objections--how-to-handle-them)
8. [Competitive Questions](#8-competitive-questions)
9. [Proof of Concept & Pilot Questions](#9-proof-of-concept--pilot-questions)
10. [Post-Sales / Implementation Questions](#10-post-sales--implementation-questions)

---

## 1. Business & ROI Questions
*Typically from: CEO, CFO, COO, VP Operations*

---

**Q1.1: What is the ROI and how quickly will we see it?**

**One-liner:** Toll fraud prevention alone typically delivers ROI within 30–60 days. PCI audit cost reduction adds $500K–$2M+ annually for mid-to-large BPOs.

**Full answer:**
ROI comes from four measurable sources:
- **Toll fraud prevention:** Average BPO loses $200K–$2M annually to SIP toll fraud. QBITEL detects patterns within 3 calls. A single prevented weekend attack (typical: $30K–$80K loss) often covers the annual license cost.
- **PCI-DSS audit scope reduction:** DTMF masking and recording controls reduce audit scope by up to 80%, cutting annual audit and compliance costs by $500K–$2M+ depending on seat count and current audit vendor.
- **SOC team efficiency:** 78% autonomous threat resolution means your security staff handle exceptions, not routine alerts. For a 5-person SOC team, this recovers 2–3 analyst hours per day.
- **Breach cost avoidance:** Average BPO data breach cost is $4.8M (IBM 2024). QBITEL prevents the insider threat and protocol-layer attacks that cause most BPO breaches.

Most customers see positive ROI within 60–90 days. We provide a pre-deployment baseline measurement so you can quantify impact directly.

---

**Q1.2: What does it cost and how is it priced?**

**One-liner:** Per concurrent agent seat, with three tiers. We size it to your actual seat count, not a generic enterprise license.

**Full answer:**
QBITEL Bridge for BPO is priced by concurrent agent seat:
- **Contact Center tier:** Up to 500 seats
- **Enterprise BPO tier:** 500–5,000 seats
- **Global BPO tier:** 5,000+ seats, unlimited tenants

All tiers include AI protocol discovery, PQC encryption, DTMF masking, toll fraud prevention, agent DLP, and compliance reporting. Higher tiers add multi-tenant management, dedicated Customer Success Manager, and SLA guarantees.

We do not charge separately for number of protocols discovered, compliance frameworks, or agent endpoints. Contact our enterprise team at enterprise@qbitel.com for a tailored quote.

---

**Q1.3: We already have a security budget committed this year. Why should we reprioritize for QBITEL?**

**One-liner:** QBITEL prevents losses that are already happening. It's not a new cost — it replaces losses you're absorbing today.

**Full answer:**
Three things are costing you money right now regardless of your security budget:
1. Toll fraud is generating fraudulent charges on your carrier bills every month — most BPOs don't attribute this correctly until auditing CDRs closely.
2. Your PCI-DSS audit prep is consuming engineering and compliance team time that could be automated.
3. Your call recordings are accumulating in a format that will be decryptable by quantum computers within the retention period required by SOX/HIPAA.

QBITEL is not an additional security cost — it is cost displacement. We recommend requesting a 2-hour Discovery Assessment (at no cost) before committing budget, so you can quantify what you're currently losing.

---

**Q1.4: Can you give us a business case template we can take to our CFO?**

**One-liner:** Yes — we provide a pre-built ROI model. We plug in your seat count, current fraud loss estimate, PCI audit spend, and SOC headcount, and it produces a 3-year NPV analysis.

**Full answer:**
Our Business Value Assessment (BVA) covers:
- Toll fraud loss baseline (we help estimate from CDR sample analysis)
- PCI-DSS audit cost reduction calculation
- SOC time recovery valuation
- Breach cost avoidance (actuarial based on BPO industry breach rates)
- License and deployment cost
- 12-month, 24-month, and 36-month ROI projection

This is typically produced during the Discovery Assessment phase and is executive-ready in PDF format. Your CFO will see numbers specific to your environment, not industry averages.

---

**Q1.5: We're not sure quantum computing is really a threat yet. Why invest now?**

**One-liner:** Your call recordings from today are being harvested now. The quantum decryption threat to future data is real, but the threat to current data is already active.

**Full answer:**
The immediate concern is not that quantum computers exist today — it's that adversaries are executing "harvest now, decrypt later" campaigns right now. They capture encrypted recordings, CDRs, and signaling data with the explicit intent to decrypt it when quantum capability arrives.

For BPOs, this is especially acute because:
- SOX requires 7-year call recording retention for financial services clients
- HIPAA requires 6-year audit retention for healthcare clients
- PCI-DSS requires recordings to be secure for the retention period

Recordings captured this week will still be within retention windows in 2031–2032 — well within the estimated quantum threat window. The time to protect them is before they are captured, not after.

Additionally, NIST finalized the first post-quantum cryptography standards in August 2024 — this is no longer experimental. Many financial services regulators are beginning to require quantum-safe roadmaps.

---

## 2. Technical & Architecture Questions
*Typically from: CTO, IT Director, Network Architect, Infrastructure Lead*

---

**Q2.1: How exactly does QBITEL deploy? Does it require agents on every endpoint?**

**One-liner:** No agents. QBITEL taps the network passively — nothing installed on PBX, agent desktops, or mainframes.

**Full answer:**
QBITEL deploys as a network-layer overlay using a passive tap on your voice and data network infrastructure. The architecture has three components:

1. **Network Sensor:** Passive tap (SPAN port or inline tap) that captures traffic for AI analysis. Read-only initially — no traffic modification during discovery phase.
2. **QBITEL Engine:** Deployed on-premise (VM or bare metal) or in your private cloud. This runs the AI discovery, PQC encryption enforcement, and agentic AI security.
3. **Management Console:** Web-based dashboard for policy management, compliance reporting, and incident response.

Nothing is installed on Avaya/Cisco/Genesys PBX systems. Nothing is installed on agent desktops for core voice protection. The optional DLP agent (for clipboard/USB/screen blocking) is a lightweight endpoint agent for agent desktops only — and even this is optional if you only want voice-layer protection.

---

**Q2.2: We run Avaya Aura with a heavily customized dial plan. Will QBITEL break anything?**

**One-liner:** No. During the discovery phase QBITEL is read-only. Encryption is activated only after you review and approve the discovered protocol map.

**Full answer:**
QBITEL's deployment follows a staged approach specifically to protect complex telephony environments:

- **Phase 1 (Passive):** Network tap placed. AI observes traffic only. Zero interference with dial plan, routing, or call flows.
- **Phase 2 (Review):** You receive a complete protocol map showing every SIP trunk, RTP stream, DTMF path, and CTI integration QBITEL discovered. Your team reviews and approves before anything is changed.
- **Phase 3 (Active):** PQC encryption is applied selectively — you choose which trunks, which call flows, which integrations to protect first. Rollback is available at any point.

We have deployed in complex Avaya environments with custom TSAPI integrations, multi-site dial plans, and legacy analog gateways. The AI learns your specific environment — it does not assume a standard configuration.

---

**Q2.3: What is the latency impact on voice quality? Our clients have strict MOS score requirements.**

**One-liner:** <2ms PQC overhead — well within the ITU-T G.114 150ms one-way delay budget. MOS scores are unaffected in all deployments to date.

**Full answer:**
Voice quality is the hardest constraint in any contact center security solution. QBITEL was built with this as a non-negotiable requirement:

- **SIP signaling encryption:** <10ms p95 including full PQC operations
- **RTP/SRTP media encryption:** <2ms per-packet overhead using ML-KEM-512 + AES-256-GCM
- **DTMF masking:** <5ms — imperceptible to callers on the far end
- **Total one-way delay budget (ITU-T G.114):** 150ms. QBITEL adds <2ms — leaving 148ms for network transit, codec processing, and jitter buffer.

We recommend including voice quality measurement (MOS, jitter, packet loss) as a specific metric in your Proof of Concept. We have never had a customer report MOS degradation attributable to QBITEL in production.

---

**Q2.4: We have a hybrid environment — some Avaya on-premise, some Genesys Cloud, some legacy analog gateways. Can QBITEL handle all of that?**

**One-liner:** Yes. QBITEL's AI discovers and protects protocol-agnostically — it doesn't require a single-vendor environment.

**Full answer:**
Hybrid telephony environments are the most common situation we encounter in large BPOs. QBITEL handles each component independently:

- **Avaya Aura on-premise:** TSAPI/DMCC integration with PQC tunnel overlay
- **Genesys Cloud:** REST API with PQC-TLS — cloud signaling secured
- **Legacy analog gateways:** Protocol-level discovery identifies analog-to-SIP conversion points; encryption applied at the SIP boundary
- **SIP trunks from multiple carriers:** Each trunk treated as a separate protection zone

The AI discovery phase maps your entire topology before any encryption is activated. You see the full picture — including connections your own team may have forgotten exist.

---

**Q2.5: We run TN3270e sessions to an IBM mainframe for customer lookups. Does QBITEL protect mainframe traffic?**

**One-liner:** Yes — TN3270e and TN5250 session protection is a core BPO module capability.

**Full answer:**
TN3270e and TN5250 terminal emulation is one of the highest-risk unprotected protocols in BPO environments. Agent sessions connecting to IBM CICS, DB2, and IMS applications typically carry full customer records — account numbers, SSNs, credit card data — over unencrypted sessions.

QBITEL provides:
- **PQC tunnel wrapping:** ML-KEM-768 + AES-256-GCM applied to all TN3270e/TN5250 sessions
- **Session monitoring:** eBPF-based detection of unusual data access patterns (bulk lookups, off-hours access)
- **Screen field detection:** Identify which screen fields contain PII for masking and audit trail purposes
- **Session audit trail:** Full per-agent, per-session record for compliance purposes

No changes required to the mainframe, CICS applications, or terminal emulator software on agent desktops.

---

**Q2.6: Our call recording platform (NICE/Verint) stores recordings in its own format. How does QBITEL interact with it?**

**One-liner:** QBITEL encrypts the RTP stream before it reaches the recording platform, and can also wrap the recording storage with quantum-safe encryption via API integration.

**Full answer:**
QBITEL operates at two points in the recording workflow:

1. **At capture time (RTP layer):** DTMF tones are masked before the audio reaches the recording platform — so card numbers never appear in recordings regardless of which recording system you use.
2. **At storage time (recording archive):** Through NICE/Verint API integration, QBITEL can enforce ML-KEM-1024 encryption on recording files and verify cryptographic integrity on retrieval.

This means:
- Recordings stored before QBITEL deployment remain as-is (we recommend a remediation plan for historical recordings)
- All recordings captured after deployment are quantum-safe
- DTMF content is masked in the audio stream, not post-processed — this is more reliable and auditor-verifiable

We have existing integration bridges for NICE CXone, NICE Engage, Verint Workforce Engagement, and Aspect WFM.

---

## 3. Security & Risk Questions
*Typically from: CISO, Security Architect, SOC Manager*

---

**Q3.1: We already have a SIEM and SOAR. How does QBITEL fit into our existing security stack?**

**One-liner:** QBITEL integrates with your SIEM via syslog/CEF and can trigger SOAR playbooks via webhooks — it adds BPO-specific context your SIEM cannot generate on its own.

**Full answer:**
QBITEL is not a replacement for your SIEM or SOAR — it is a specialized sensor and response layer for BPO-specific threats that generic security tools cannot detect.

Integration options:
- **SIEM:** CEF/syslog output to Splunk, QRadar, Microsoft Sentinel, ArcSight. All QBITEL events include BPO-specific context (call_id, agent_id, tenant_id, trunk_id, fraud_type).
- **SOAR:** Webhook triggers to Palo Alto XSOAR, Splunk SOAR, ServiceNow SecOps. QBITEL can trigger playbooks or receive containment instructions.
- **Threat Intelligence:** IOC feeds from QBITEL's toll fraud database (200+ country premium-rate prefix database) can be exported to your TI platform.

What your existing SIEM cannot do that QBITEL provides:
- Correlate security events with call-level data (which agent, which call, which trunk)
- Detect SIP-specific attacks (INVITE flooding, toll fraud, DTMF interception)
- Enforce PCI-DSS DTMF masking at the protocol layer
- Provide per-tenant compliance isolation for multi-tenant environments

---

**Q3.2: How does QBITEL's AI make autonomous response decisions? What prevents it from taking the wrong action?**

**One-liner:** Four safety controls: confidence thresholds, risk classification, blast radius limits, and human override. Any action below confidence or above risk threshold escalates to your SOC.

**Full answer:**
QBITEL's autonomous response framework has explicit safety boundaries:

**Decision matrix:**
| Confidence | Low Risk | Medium Risk | High Risk |
|------------|----------|-------------|-----------|
| High (95%+) | Auto-execute | Auto-approve | Escalate |
| Medium (85–95%) | Auto-execute | Escalate | Escalate |
| Low (<85%) | Escalate | Escalate | Escalate |

**Safety constraints hardcoded into the system:**
- **Blast radius limit:** No autonomous action affects more than 10 systems without human approval
- **Production protection:** Actions on systems tagged as production-critical require human approval regardless of confidence
- **Rollback window:** All autonomous actions are reversible within 60 minutes with a single click
- **Audit trail:** Every automated action generates a full audit log with the LLM reasoning chain, confidence score, evidence used, and action taken
- **Emergency stop:** Physical and software-based emergency stop that freezes all autonomous actions immediately

Your SOC can review and tune confidence thresholds per action type. You can also set the system to "alert only" mode for any action category where you prefer human decision-making.

---

**Q3.3: Where does the LLM reasoning run? Is our call data going to OpenAI or Anthropic?**

**One-liner:** On-premise by default — Ollama running Llama 3 or Mixtral on your infrastructure. No call data ever leaves your network.

**Full answer:**
QBITEL's agentic AI reasoning runs entirely on-premise using Ollama as the LLM serving layer. Supported models:
- Llama 3.2 (8B for fast response, 70B for complex analysis)
- Mixtral 8x7B (for reasoning tasks)
- Qwen 2.5 (for multilingual BPO environments)
- Phi-3 (lightweight, for resource-constrained deployments)

The on-premise LLM has access only to anonymized security event data — it does not process raw call audio, customer PII, or cardholder data. Event data never leaves the QBITEL Engine VM.

Optional cloud LLM (Claude API) is available as an upgrade for customers who require the most advanced reasoning capability and have data sovereignty approvals for cloud processing. This is opt-in and never the default.

For air-gapped environments (government BPOs, classified-adjacent operations): QBITEL supports fully disconnected deployment with on-premise model serving — no internet connectivity required after initial deployment.

---

**Q3.4: How does QBITEL handle zero-day SIP vulnerabilities?**

**One-liner:** The AI detects behavioral anomalies independent of signatures — zero-days that exhibit unusual call patterns, volume spikes, or protocol deviations are flagged without needing a signature update.

**Full answer:**
Traditional SIP security tools rely on signature-based detection — they can only detect known attacks. QBITEL uses behavioral anomaly detection that identifies attacks based on deviation from learned baseline behavior, making it effective against zero-days.

Specifically for SIP:
- **Grammar-based validation:** QBITEL has learned the exact structure of SIP messages on your network. Messages that deviate from learned grammar (malformed headers, unusual field values, unexpected message sequences) are flagged — even if the attack pattern is new.
- **Volume anomaly detection:** Sudden spikes in INVITE rates, unusual destination patterns, or off-hours call volumes are detected without requiring a signature match.
- **State machine enforcement:** SIP call state transitions (INVITE → 100 Trying → 180 Ringing → 200 OK) are enforced. Out-of-sequence messages trigger immediate analysis.
- **Threat intelligence correlation:** The 200+ country premium-rate prefix database is updated continuously and correlated against all outbound call destinations in real time.

Signature updates are still delivered for known threats, but they are not the primary detection mechanism.

---

**Q3.5: What happens if QBITEL itself is compromised? Are we creating a new single point of failure?**

**One-liner:** QBITEL is designed with fail-open voice and fail-closed security — voice calls continue if QBITEL has an issue; security enforcement defaults to blocking unknown traffic.

**Full answer:**
QBITEL's resilience architecture addresses this directly:

- **Fail-open voice path:** If the QBITEL Engine becomes unavailable, the network tap can be bypassed so voice calls continue uninterrupted. You lose security enforcement temporarily but not call operations.
- **High availability:** QBITEL Engine supports active-active clustering with automatic failover. For large deployments, we recommend N+1 redundancy.
- **Immutable audit logs:** All audit trails are written to tamper-evident storage independent of the QBITEL Engine — even if the engine is compromised, forensic history is preserved.
- **Cryptographic key isolation:** Encryption keys are stored in an HSM (Hardware Security Module) — compromise of the QBITEL Engine does not expose keys.
- **Network segmentation:** QBITEL Engine is deployed in an isolated management network segment with no direct access to agent endpoints or PBX systems.
- **Supply chain integrity:** All QBITEL components are signed and verified on every startup. Tampered binaries will not load.

We are happy to share our threat model document during the technical evaluation phase.

---

## 4. Compliance & Legal Questions
*Typically from: Chief Compliance Officer, Data Protection Officer, Legal Counsel*

---

**Q4.1: We're undergoing a PCI-DSS 4.0 audit next quarter. Can QBITEL help us pass it?**

**One-liner:** Yes — QBITEL directly addresses Requirements 3, 4, 7, 8, and 12 of PCI-DSS 4.0. We can generate audit-ready evidence packages.

**Full answer:**
QBITEL maps to PCI-DSS 4.0 requirements specifically for voice channel environments:

| PCI-DSS 4.0 Requirement | QBITEL Capability |
|--------------------------|-------------------|
| Req 3: Protect stored cardholder data | ML-KEM-1024 recording encryption; DTMF data never stored |
| Req 4: Protect cardholder data in transit | PQC-TLS for SIP; SRTP-PQC for RTP; quantum-safe tunnels |
| Req 7: Restrict access to cardholder data | Agent screen masking; per-role access controls |
| Req 8: Identify users and authenticate | MFA enforcement; session validation |
| Req 9: Restrict physical access | Remote agent endpoint compliance |
| Req 12: Support security with policies | Automated policy enforcement and evidence logging |

For your upcoming audit specifically:
- QBITEL generates a PCI-DSS evidence package covering voice channel controls
- DTMF masking events are logged with timestamp, agent_id, call_id, and masking mode — directly usable as audit evidence
- Recording encryption keys and their rotation history are exportable for auditor review
- Scope reduction documentation is auto-generated based on actual call flow analysis

We recommend deploying at least 60 days before your audit date to establish a baseline of evidence. If your audit is sooner, contact us for an expedited deployment path.

---

**Q4.2: We serve both EU clients (GDPR) and US financial clients (PCI-DSS, SOX). How does QBITEL handle overlapping and sometimes conflicting compliance requirements?**

**One-liner:** Per-tenant compliance policies — each client's data is governed by their specific framework with full isolation between tenants.

**Full answer:**
Multi-framework compliance in a multi-tenant BPO is QBITEL's specific design target. The architecture:

- **Per-tenant policy engine:** Each enterprise client gets a separate compliance policy (PCI-DSS, GDPR, HIPAA, SOX — or any combination). Policies are enforced independently, not averaged.
- **Per-tenant encryption keys:** Cryptographic isolation ensures that a legal hold or data request for one client cannot access another client's data.
- **Conflicting requirements handling:** Where GDPR (right to erasure) conflicts with SOX (7-year retention), QBITEL applies the client-specific policy — the BPO does not need to resolve the legal conflict; it enforces what the contract specifies.
- **Jurisdictional data routing:** For GDPR clients, QBITEL can enforce data residency — recording metadata and call data stay in EU-hosted storage, while QBITEL Engine applies policies without storing PII.
- **Independent reporting:** Each client receives a compliance report for their specific framework(s) — no cross-tenant data in any report.

We recommend involving your Data Protection Officer and client contract leads in the policy configuration phase to ensure alignment with contractual obligations.

---

**Q4.3: GDPR requires us to have a lawful basis for recording calls. Can QBITEL help with consent management?**

**One-liner:** Yes — QBITEL includes consent tracking and can enforce conditional recording based on consent status.

**Full answer:**
QBITEL's GDPR module for BPOs includes:
- **Consent event logging:** Recording consent (IVR prompt response, agent-captured consent) is logged with timestamp and call_id — linked to the recording file.
- **Conditional recording:** If consent is withdrawn mid-call, QBITEL can automatically pause or terminate recording with an audit event.
- **DSAR support:** Data Subject Access Requests — QBITEL can locate all recordings, call metadata, and agent screen events associated with a specific caller phone number or customer ID.
- **Right to erasure:** QBITEL can cryptographically shred (key deletion) recordings associated with a specific data subject, satisfying GDPR Article 17 without requiring physical deletion from archive systems.
- **Retention enforcement:** Automatic deletion or archival workflows triggered by configurable retention periods per client and per data category.

Consent management integration is available for common IVR platforms (Nuance, Genesys Dialog Engine, Amazon Connect, NICE CXone) and can be triggered via REST API from any custom consent management system.

---

**Q4.4: We need to demonstrate HIPAA compliance for a healthcare BPO contract. What documentation can QBITEL provide?**

**One-liner:** QBITEL generates a HIPAA Technical Safeguards evidence package covering Administrative, Physical, and Technical safeguard requirements automatically.

**Full answer:**
For healthcare BPO specifically, QBITEL covers HIPAA Technical Safeguards (45 CFR § 164.312):

| HIPAA Technical Safeguard | QBITEL Evidence |
|---------------------------|----------------|
| Access Control (§164.312(a)) | Agent session logs, role-based access enforcement |
| Audit Controls (§164.312(b)) | Immutable per-call, per-agent, per-action audit trail |
| Integrity (§164.312(c)) | Cryptographic integrity verification on all PHI-adjacent recordings |
| Transmission Security (§164.312(e)) | PQC encryption on all voice, terminal, and data transmissions |

QBITEL also supports HIPAA's Minimum Necessary standard by flagging agents who access patient records beyond their assigned scope (e.g., a billing agent looking up clinical data).

For your healthcare client contract specifically, we can provide:
- A HIPAA Technical Safeguards attestation document
- Sample Business Associate Agreement (BAA) language covering QBITEL's role
- 6-year audit log retention configuration matching HIPAA requirements

---

## 5. Operations Questions
*Typically from: Contact Center Director, VP Operations, Workforce Management*

---

**Q5.1: Will QBITEL affect our agents in any way? We can't retrain 5,000 people.**

**One-liner:** Zero change for agents on voice protection. Optional DLP controls (clipboard blocking, watermarking) are configured per policy — agents see no difference in how they work.

**Full answer:**
Agent experience is a deployment constraint we take seriously. Impact by component:

| QBITEL Component | Agent Impact |
|-----------------|--------------|
| Voice PQC encryption | Zero — transparent at network layer |
| DTMF masking | Zero for agent; caller hears normal tones |
| Toll fraud prevention | Zero unless their call is blocked for fraud |
| Recording encryption | Zero — recording interface unchanged |
| TN3270e session protection | Zero — terminal emulator unchanged |
| Screen watermarking | Visible watermark in corner if configured; invisible option available |
| Clipboard DLP | Agents cannot paste PII to unauthorized apps — may require workflow adjustment for ~2% of use cases |
| USB blocking | USB drives blocked — agents need to use approved file transfer methods |

For the DLP controls that do affect agent workflow (clipboard, USB), we recommend a phased rollout starting with a pilot group, collecting feedback, and adjusting policy before full rollout. We provide a change management guide for operations teams.

No retraining required for core voice security. DLP policy rollout typically requires a 30-minute "what changed and why" briefing for agents — not a training program.

---

**Q5.2: We run 24/7 operations across multiple time zones. What is the impact of deployment on live operations?**

**One-liner:** Zero downtime deployment. The network tap is placed during off-peak hours if preferred, but it does not require any call interruption even during business hours.

**Full answer:**
QBITEL's deployment architecture was designed for 24/7 contact center environments:

- **Network tap placement:** A passive tap (SPAN port or physical tap) is placed on the network switch. This does not interrupt traffic — it mirrors traffic to the QBITEL sensor.
- **Discovery phase:** Entirely passive — no traffic modification. Can run during live operations indefinitely.
- **Encryption activation:** Done per-trunk or per-protocol on a schedule you control. Typically done site-by-site, starting with a single trunk in each location.
- **Rollback:** If any issue is detected, encryption can be disabled for a specific trunk in under 60 seconds from the management console without affecting other trunks.
- **Emergency bypass:** Hardware bypass is available that removes the QBITEL sensor from the network path entirely in under 5 seconds if required.

For a 5,000-seat BPO across 3 sites, a typical deployment schedule is:
- Day 1: Tap installation at all sites (passive, zero impact)
- Day 2–5: Discovery phase (passive, zero impact)
- Week 2: Review protocol map and approve activation plan
- Week 2–3: Activate one trunk per site, monitor for 48 hours each
- Week 3–4: Full activation across all trunks

---

**Q5.3: We have remote agents in India, Philippines, and Eastern Europe. Does QBITEL support international remote deployments?**

**One-liner:** Yes — the remote agent tunnel is location-agnostic. Geo-fencing can restrict or allow specific countries per your policy.

**Full answer:**
QBITEL's remote agent security is designed for globally distributed BPO workforces:

- **Tunnel technology:** ML-KEM-768 + AES-256-GCM hybrid tunnels work regardless of agent location. Performance is optimized for high-latency connections (tested at 150ms+ base latency without quality degradation).
- **Geo-fencing options:**
  - Allow specific countries only (e.g., India, Philippines, US — block all others)
  - Alert on unexpected location changes (agent normally in Bangalore, logging in from Lagos)
  - Client-specific rules (banking client agents must be in approved countries only)
- **Home network assessment:** Agents' home WiFi assessed for WPA2/WPA3 compliance. Agents on WEP or open networks are blocked from connecting.
- **ISP-level controls:** Can restrict to approved ISP ranges if required by high-security clients.
- **MPLS/SD-WAN integration:** For BPOs with managed WAN to offshore sites, QBITEL integrates with SD-WAN fabric for consistent policy enforcement.

For India and Philippines specifically — the two largest BPO locations globally — we have reference deployments with configuration guides optimized for common ISP environments (Jio, Airtel, PLDT, Globe).

---

**Q5.4: Our IVR system handles payment card entry by DTMF. How does QBITEL affect the IVR payment flow?**

**One-liner:** QBITEL protects the DTMF tones between IVR and agent/recording without changing the IVR application logic at all.

**Full answer:**
IVR payment flows have two modes, and QBITEL handles both:

**Mode 1: Agent-assisted payment (agent hears DTMF tones)**
- QBITEL masks DTMF tones in the agent's audio stream and in the recording
- The payment gateway still receives the DTMF signals correctly (QBITEL does not interfere with the SIP signaling to the payment gateway)
- Agent hears silence or flat tone during card entry — they cannot transcribe the digits even if they wanted to

**Mode 2: Pure IVR payment (no agent, caller enters card via IVR)**
- QBITEL monitors the IVR session for PAN detection
- DTMF input is protected in transit between caller and IVR system
- Recording of IVR session is encrypted and DTMF tones are masked in recording
- If the IVR sends cardholder data to a backend CRM/ticketing system, QBITEL monitors that data path for PAN leakage

In both cases, the IVR application (Nuance, Genesys, Cisco CVP, Avaya IR, VXML) is not modified. The protection operates at the SIP/RTP protocol layer beneath the application.

---

## 6. Procurement & Vendor Questions
*Typically from: Procurement, Vendor Management, Legal*

---

**Q6.1: How do we know QBITEL will still be around in 5 years? You're not a Cisco or Palo Alto.**

**One-liner:** The same way you evaluate any vendor: architecture, customer base, financial stability, and contractual protections. We offer source code escrow for enterprise contracts.

**Full answer:**
Legitimate concern for any new vendor. Here is how we address it:

- **Source code escrow:** For Enterprise and Global tier contracts, we offer source code escrow with a reputable escrow provider. If QBITEL ceases operations, you receive the source code to operate the platform independently.
- **Open standards:** QBITEL uses NIST-standardized algorithms (ML-KEM, ML-DSA) and open protocols (SIP, SRTP, TN3270e). If we disappear, the encryption keys and data formats are not proprietary — you are not locked into our decryption capability.
- **Customer references:** We provide customer references in your industry segment for verification calls.
- **Financial disclosure:** We share financial stability information under NDA for enterprise procurement processes.
- **Contractual protections:** Our enterprise contracts include SLA guarantees, data portability provisions, and exit assistance clauses.

We also encourage you to evaluate the alternative risk: using no solution for a threat (quantum harvesting, toll fraud) that is real today. The cost of inaction has a probability and a dollar value.

---

**Q6.2: What does your data processing agreement look like? Who is the data controller vs. processor?**

**One-liner:** QBITEL acts as a data processor. Your BPO is the data controller. We can sign your DPA or provide our standard DPA — both are available.

**Full answer:**
QBITEL's data processing role is limited to security event processing — we do not process customer PII for any purpose other than the security service you contracted.

Standard DPA terms:
- **Data processor role:** QBITEL processes security event data (call metadata, protocol data, agent behavior signals) as processor on behalf of your BPO
- **Sub-processors:** Disclosed list of infrastructure sub-processors (cloud provider for SaaS components if applicable). On-premise deployments have no sub-processors.
- **Data retention:** Security event data retained for 90 days by default, configurable. Compliance audit logs retained per your configured retention policy.
- **Data transfers:** For on-premise deployments, no data leaves your infrastructure. For any cloud management console usage, data transfer agreements per GDPR Chapter V are in place.
- **Breach notification:** QBITEL commits to 24-hour notification of any security incident affecting your data.

We are happy to review your standard vendor DPA and markup within our legal team's standard turnaround time.

---

**Q6.3: We need to do a security assessment of QBITEL before we can onboard you as a vendor. What do you provide?**

**One-liner:** We provide SOC 2 Type II report, penetration test summary, SBOM, architecture threat model, and support customer-led penetration testing.

**Full answer:**
QBITEL supports enterprise vendor security assessment with the following artifacts:

- **SOC 2 Type II report:** Available under NDA covering Security, Availability, and Confidentiality trust service criteria
- **Penetration test report:** Annual third-party penetration test summary (full report under NDA)
- **Software Bill of Materials (SBOM):** Complete SBOM for vulnerability tracking
- **Architecture threat model:** Formal threat model document covering all system components and mitigations
- **CVE response policy:** Our commitment to patch critical CVEs within 24 hours, high within 72 hours
- **Customer-led pen testing:** We support your security team or contracted pen testers performing assessment of the QBITEL deployment in your environment — with appropriate coordination to avoid false positive autonomous responses

For vendor questionnaires (SIG, CAIQ, custom), our vendor security team responds within 5 business days.

---

## 7. Hard Objections & How to Handle Them
*The toughest pushback you will face in the room*

---

**Objection 7.1: "We already have Palo Alto / Fortinet / CrowdStrike. Why do we need another security product?"**

**Response approach:** Acknowledge their investment, then expose the gap.

"Your existing tools do excellent work protecting what they know about — endpoints, known malware, network perimeter threats. None of them were designed for what happens inside a contact center at the protocol layer.

Ask your team: Can Palo Alto tell you whether a DTMF tone in a call recording is a card number? Can CrowdStrike detect when an agent is reading card numbers aloud? Can Fortinet identify IRSF fraud in SIP CDRs within 3 calls?

These are BPO-specific problems that require BPO-specific solutions. We are not a replacement for your existing stack — we are the layer that covers the 30% of your risk that your existing stack cannot see."

---

**Objection 7.2: "We don't want to add complexity to our environment. We're already managing too many vendors."**

**Response approach:** Frame QBITEL as a consolidation play, not an addition.

"Understood. Let me ask — how many separate tools are you currently using to address DTMF compliance, toll fraud monitoring, remote agent security, and compliance reporting? Most BPOs we talk to have 3–5 point solutions for these problems, or manual processes.

QBITEL replaces all of those with a single platform. For many customers, deploying QBITEL allows them to retire their separate toll fraud monitoring tool, their manual PCI audit prep process, and their VPN infrastructure for remote agents. The net result is usually fewer vendors, not more."

---

**Objection 7.3: "The quantum threat is overstated. We'll deal with it when it's real."**

**Response approach:** Shift the frame from future threat to current attack.

"I agree quantum computers aren't breaking encryption today. But the attack that's happening right now is the harvest — adversaries are collecting your encrypted traffic today to decrypt later.

Here's the question that matters: Are your call recordings from this year covered by a 7-year SOX retention policy? If yes, those recordings will still exist in 2031. NIST's estimate for cryptographically relevant quantum computers is 2030–2035.

The recordings you capture today are the ones at risk — not recordings from some future date after you've had time to prepare. Waiting means you're already too late for the data you're generating right now."

---

**Objection 7.4: "We tried a security product like this before and it broke our voice quality. We won't take that risk again."**

**Response approach:** Acknowledge the trauma, offer proof-first approach.

"That's a fair and important concern, and I want to take it seriously. Voice quality is non-negotiable in a contact center — I completely understand why that experience has made you cautious.

Here's what I'd propose: our Discovery Assessment phase is entirely passive — no traffic modification, zero risk to voice quality. Once you've seen the protocol map and understand exactly what we do, we would start with a single, low-risk trunk in a test environment or a non-critical location.

We also ask you to include MOS score measurement as a specific metric in any proof of concept. If we degrade your MOS scores at all, you stop the deployment and pay nothing further. We're confident enough in our <2ms overhead guarantee to put that in writing."

---

**Objection 7.5: "We're in the middle of a platform migration to [Genesys Cloud / Amazon Connect / NICE CXone]. We'll evaluate security solutions after the migration."**

**Response approach:** Position QBITEL as migration insurance, not post-migration addition.

"Migrations are exactly when BPOs are most vulnerable. During the transition period, you typically have both old and new infrastructure running simultaneously — creating a complex, partially-monitored environment that attackers love.

QBITEL deploys protocol-agnostically. It protects your legacy Avaya traffic today and your new Genesys Cloud traffic tomorrow without requiring two separate deployments. You get consistent security coverage throughout the migration, and you arrive at your new platform already protected rather than having to bolt security on afterward.

We've worked with several BPOs through platform migrations specifically for this reason. Would it help to speak with one of them?"

---

**Objection 7.6: "We need board approval for any security spend over $X. This will take 6 months."**

**Response approach:** Offer a path that starts without board approval.

"Understood — we work within enterprise procurement realities all the time. Here's an alternative path:

Our Discovery Assessment is at no cost and requires no procurement process. It gives you a written report on what's running on your network, what's unencrypted, and what your current fraud loss exposure looks like.

That report is often exactly what accelerates board approval — because it quantifies a risk that currently has no number attached to it. A board can evaluate 'approve $X to prevent a documented $Y annual loss' much faster than 'approve $X for a security enhancement.'

Can we schedule the Discovery Assessment now, with the understanding that it's the input to your board presentation rather than the approval outcome?"

---

## 8. Competitive Questions
*When prospects ask about specific alternatives*

---

**Q8.1: We're evaluating Pindrop for voice fraud detection. How is QBITEL different?**

**One-liner:** Pindrop detects voice biometric fraud (caller identity spoofing). QBITEL protects the protocol infrastructure — SIP encryption, toll fraud in CDRs, agent desktop DLP, compliance. Complementary, not competing.

**Full answer:**
Pindrop is a caller authentication and voice biometric solution — it answers "Is this caller who they claim to be?" QBITEL answers "Is the network, the agent, and the compliance posture secure?" These are different problem spaces.

Specifically, QBITEL covers what Pindrop does not:
- SIP/RTP encryption (Pindrop does not encrypt voice channels)
- Toll fraud detection in CDR patterns (IRSF, PBX hacking — Pindrop does caller authentication, not trunk-level fraud)
- Agent desktop DLP (not in Pindrop scope)
- PCI-DSS DTMF masking (not in Pindrop scope)
- Post-quantum cryptography (not in Pindrop scope)
- Compliance reporting across 9 frameworks (not in Pindrop scope)

Many BPOs run both — Pindrop for caller authentication, QBITEL for infrastructure and compliance security.

---

**Q8.2: Our parent company uses Cisco security across all their businesses. Can we just extend that?**

**One-liner:** Cisco's security stack does not include BPO-specific controls: DTMF masking, toll fraud CDR analysis, TN3270e session protection, or multi-tenant compliance isolation.

**Full answer:**
Cisco's portfolio (SecureX, Talos, Umbrella, Firepower) is strong for network perimeter and endpoint security. For BPO-specific requirements, the gaps are significant:

- Cisco does not offer DTMF masking for call recordings (a PCI-DSS requirement specific to voice payment channels)
- Cisco Talos handles threat intelligence but does not analyze CDR patterns for toll fraud (IRSF, Wangiri, call pumping)
- Cisco does not protect TN3270e/TN5250 mainframe sessions specifically
- Cisco does not offer multi-tenant compliance isolation (separate PCI-DSS, HIPAA, SOC 2 policies per client)
- Cisco does not offer post-quantum cryptography for SIP/RTP channels as a production feature (as of 2026)

We often deploy alongside Cisco infrastructure — QBITEL integrates with Cisco CUCM and Finesse and can feed events to Cisco XDR/SecureX. It is not an either/or decision.

---

**Q8.3: We looked at a solution from [insert telecom vendor] that's bundled with our SIP trunks. Why pay separately?**

**One-liner:** Carrier-bundled security is limited to carrier-visible threats on carrier-owned infrastructure. It does not protect your internal PBX, agent desktops, mainframe sessions, or multi-tenant compliance posture.

**Full answer:**
Carrier-bundled security is attractive on paper but limited in practice because carriers can only protect what they own:

- Carrier sees your SIP trunks — not your internal PBX, not your TN3270e sessions, not your agent desktops
- Carrier fraud detection operates on 72-hour billing cycles for most providers — by the time you're notified, the fraud has run for days
- Carrier cannot enforce DTMF masking inside your recording system
- Carrier cannot monitor agent behavior or DLP
- Carrier cannot generate PCI-DSS or HIPAA compliance evidence for your auditors
- Carrier-bundled solutions have no incentive to block calls aggressively — that reduces their revenue

QBITEL detects toll fraud within 3 calls (seconds to minutes), not on the billing cycle. And it covers the 90% of your security posture that lives inside your perimeter, not on the carrier network.

---

## 9. Proof of Concept & Pilot Questions
*When prospects are ready to test but have specific requirements*

---

**Q9.1: What does a PoC look like and what do we need to commit to run one?**

**One-liner:** 2 weeks, 1 network tap, 1 trunk for active testing. You provide a network engineer for 4 hours of setup; we do everything else.

**Full answer:**
Standard QBITEL BPO Proof of Concept:

**Duration:** 2 weeks (can be extended to 30 days)

**Your commitment:**
- Network engineer: ~4 hours for tap placement and network access
- Security/compliance stakeholder: 2 hours for policy configuration review
- Operations: Designation of 1 test trunk and 20–50 agent sessions for active testing

**What QBITEL delivers:**
- Complete protocol map of your environment
- Toll fraud detection live on test trunk
- DTMF masking demo on test calls
- PCI-DSS evidence report for the test period
- Compliance dashboard with 90-day projected compliance posture
- Performance report: MOS scores before and after, latency measurements
- Estimated annual ROI based on observed fraud patterns

**Success criteria:** You define them. We recommend: (a) zero MOS degradation, (b) at least one fraud pattern detected, (c) PCI evidence package generated, (d) deployment completed in <6 hours.

**Cost:** No charge for PoC. If you proceed to production, PoC costs are credited toward first-year license.

---

**Q9.2: Can we test with production traffic or does it need to be a lab environment?**

**One-liner:** Production traffic preferred — that's where real fraud patterns live. The discovery phase is passive and safe for production from day one.

**Full answer:**
A lab environment will not give you meaningful results for two critical use cases:
1. **Toll fraud detection:** Fraud patterns only appear in production CDR data — you cannot simulate IRSF or Wangiri patterns in a lab environment meaningfully.
2. **Protocol discovery accuracy:** The AI needs to see your actual traffic to learn your environment. Lab traffic is typically too clean and structured to reveal the edge cases.

QBITEL's phased deployment is specifically designed to be safe for production:
- Passive tap phase: Zero traffic modification — 100% safe for production from day one
- Protocol discovery: Read-only analysis of production traffic
- Active encryption: Applied to one designated test trunk only, leaving all others untouched

We recommend using production traffic with a single low-risk trunk (e.g., an outbound dialer trunk or an internal test trunk) for the active phase.

---

## 10. Post-Sales / Implementation Questions
*When the deal is done and implementation begins*

---

**Q10.1: What does the support model look like after deployment?**

**One-liner:** 24/7 technical support for Enterprise and Global tiers, with a dedicated Customer Success Manager for the first 90 days and ongoing quarterly reviews.

**Full answer:**
Post-deployment support structure:

| Tier | Support | CSM | Response SLA |
|------|---------|-----|-------------|
| Contact Center | Business hours (8/5), email + ticket | Shared | P1: 4 hours |
| Enterprise BPO | 24/7, phone + email + ticket | Dedicated 90 days | P1: 1 hour |
| Global BPO | 24/7, phone + email + dedicated Slack | Dedicated ongoing | P1: 30 min |

**What a P1 (Critical) event looks like for BPO:**
- Voice quality degradation attributable to QBITEL
- Autonomous response action that affected live call operations
- False positive fraud block preventing legitimate calls

For all tiers, the first 90 days include:
- Weekly check-in calls
- Model tuning based on your environment's baseline
- False positive review and policy refinement
- Compliance report review and auditor preparation support

---

**Q10.2: How do we keep QBITEL updated? What is the patch/upgrade process?**

**One-liner:** Automated for security signatures (toll fraud database, threat intelligence). Manual approval for engine updates. Zero downtime for signature updates.

**Full answer:**
QBITEL has two update categories:

**Automatic (no downtime, no approval required):**
- Toll fraud prefix database (updated daily — new premium-rate numbers added globally)
- Threat intelligence feeds (updated every 4 hours)
- ML model signature updates (weekly, applied to inference layer without engine restart)

**Manual (requires your approval, zero-downtime rolling update):**
- QBITEL Engine software versions (quarterly feature releases)
- PQC algorithm updates (triggered by NIST updates or new vulnerabilities)
- Protocol parser updates (for newly discovered protocol variants in your environment)

For on-premise deployments: Updates are delivered as signed packages to your QBITEL Engine. Your team controls the installation schedule. We recommend a maintenance window for engine updates, but it is not mandatory — rolling updates keep at least one node active at all times in clustered deployments.

For the LLM models (Ollama/on-premise): Model updates are optional and evaluated by your team. We provide release notes on capability changes before any model update.

---

*Document Version 1.0 | February 2026*
*QBITEL Bridge — BPO Sales Enablement | Confidential — Internal Use Only*
*enterprise@qbitel.com | https://bridge.qbitel.com*
